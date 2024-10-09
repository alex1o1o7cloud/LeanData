import Mathlib

namespace last_person_teeth_removed_l245_24533

-- Define the initial conditions
def total_teeth : ℕ := 32
def total_removed : ℕ := 40
def first_person_removed : ℕ := total_teeth * 1 / 4
def second_person_removed : ℕ := total_teeth * 3 / 8
def third_person_removed : ℕ := total_teeth * 1 / 2

-- Express the problem in Lean
theorem last_person_teeth_removed : 
  first_person_removed + second_person_removed + third_person_removed + last_person_removed = total_removed →
  last_person_removed = 4 := 
by
  sorry

end last_person_teeth_removed_l245_24533


namespace wall_area_in_square_meters_l245_24568

variable {W H : ℤ} -- We treat W and H as integers referring to centimeters

theorem wall_area_in_square_meters 
  (h₁ : W / 30 = 8) 
  (h₂ : H / 30 = 5) : 
  (W / 100) * (H / 100) = 360 / 100 :=
by 
  sorry

end wall_area_in_square_meters_l245_24568


namespace solution_set_of_inequality_l245_24513

theorem solution_set_of_inequality (x : ℝ) : (∃ x, (0 ≤ x ∧ x < 1) ↔ (x-2)/(x-1) ≥ 2) :=
sorry

end solution_set_of_inequality_l245_24513


namespace triangle_perimeter_l245_24581

theorem triangle_perimeter (a b : ℝ) (x : ℝ) 
  (h₁ : a = 3) 
  (h₂ : b = 5) 
  (h₃ : x ^ 2 - 5 * x + 6 = 0)
  (h₄ : 2 < x ∧ x < 8) : a + b + x = 11 :=
by sorry

end triangle_perimeter_l245_24581


namespace point_on_graph_l245_24545

def lies_on_graph (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  y = f x

theorem point_on_graph :
  lies_on_graph (-2) 0 (λ x => (1 / 2) * x + 1) :=
by
  sorry

end point_on_graph_l245_24545


namespace unique_two_digit_factors_l245_24584

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def factors (n : ℕ) (a b : ℕ) : Prop := a * b = n

theorem unique_two_digit_factors : 
  ∃! (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors 1950 a b :=
by sorry

end unique_two_digit_factors_l245_24584


namespace remaining_credit_to_be_paid_l245_24529

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l245_24529


namespace bowling_tournament_l245_24541

-- Definition of the problem conditions
def playoff (num_bowlers: Nat): Nat := 
  if num_bowlers < 5 then
    0
  else
    2^(num_bowlers - 1)

-- Theorem statement to prove
theorem bowling_tournament (num_bowlers: Nat) (h: num_bowlers = 5): playoff num_bowlers = 16 := by
  sorry

end bowling_tournament_l245_24541


namespace tournament_chromatic_index_l245_24553

noncomputable def chromaticIndex {n : ℕ} (k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) : ℕ :=
k

theorem tournament_chromatic_index (n k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) :
  chromaticIndex k h₁ h₂ = k :=
by sorry

end tournament_chromatic_index_l245_24553


namespace roots_squared_sum_eq_13_l245_24562

/-- Let p and q be the roots of the quadratic equation x^2 - 5x + 6 = 0. Then the value of p^2 + q^2 is 13. -/
theorem roots_squared_sum_eq_13 (p q : ℝ) (h₁ : p + q = 5) (h₂ : p * q = 6) : p^2 + q^2 = 13 :=
by
  sorry

end roots_squared_sum_eq_13_l245_24562


namespace both_questions_correct_l245_24517

def total_students := 100
def first_question_correct := 75
def second_question_correct := 30
def neither_question_correct := 20

theorem both_questions_correct :
  (first_question_correct + second_question_correct - (total_students - neither_question_correct)) = 25 :=
by
  sorry

end both_questions_correct_l245_24517


namespace prove_mutually_exclusive_and_exhaustive_events_l245_24511

-- Definitions of conditions
def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2

-- Definitions of options
def option_A : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ ¬b3 ∧ ¬g1 ∧ g2)  -- Exactly 1 boy and exactly 2 girls
def option_B : Prop := (∃ (b1 b2 b3 : Bool), b1 ∧ b2 ∧ b3)  -- At least 1 boy and all boys
def option_C : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ (b3 ∨ g1 ∨ g2))  -- At least 1 boy and at least 1 girl
def option_D : Prop := (∃ (b1 b2 : Bool) (g3 : Bool), b1 ∧ ¬b2 ∧ g3)  -- At least 1 boy and all girls

-- The proof statement showing that option_D == Mutually Exclusive and Exhaustive Events
theorem prove_mutually_exclusive_and_exhaustive_events : option_D :=
sorry

end prove_mutually_exclusive_and_exhaustive_events_l245_24511


namespace find_average_speed_l245_24580

noncomputable def average_speed (distance1 distance2 : ℝ) (time1 time2 : ℝ) : ℝ := 
  (distance1 + distance2) / (time1 + time2)

theorem find_average_speed :
  average_speed 1000 1000 10 4 = 142.86 := by
  sorry

end find_average_speed_l245_24580


namespace count_library_books_l245_24557

theorem count_library_books (initial_library_books : ℕ) 
  (books_given_away : ℕ) (books_added_from_source : ℕ) (books_donated : ℕ) 
  (h1 : initial_library_books = 125)
  (h2 : books_given_away = 42)
  (h3 : books_added_from_source = 68)
  (h4 : books_donated = 31) : 
  initial_library_books - books_given_away - books_donated = 52 :=
by sorry

end count_library_books_l245_24557


namespace clarissa_copies_needed_l245_24535

-- Define the given conditions
def manuscript_pages : ℕ := 400
def cost_per_page : ℚ := 0.05
def cost_per_binding : ℚ := 5.00
def total_cost : ℚ := 250.00

-- Calculate the total cost for one manuscript
def cost_per_copy_and_bind : ℚ := cost_per_page * manuscript_pages + cost_per_binding

-- Define number of copies needed
def number_of_copies_needed : ℚ := total_cost / cost_per_copy_and_bind

-- Prove number of copies needed is 10
theorem clarissa_copies_needed : number_of_copies_needed = 10 := 
by 
  -- Implementing the proof steps would go here
  sorry

end clarissa_copies_needed_l245_24535


namespace f_m_plus_1_positive_l245_24523

def f (a x : ℝ) := x^2 + x + a

theorem f_m_plus_1_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := 
  sorry

end f_m_plus_1_positive_l245_24523


namespace triangle_area_calculation_l245_24563

theorem triangle_area_calculation
  (A : ℕ)
  (BC : ℕ)
  (h : ℕ)
  (nine_parallel_lines : Bool)
  (equal_segments : Bool)
  (largest_area_part : ℕ)
  (largest_part_condition : largest_area_part = 38) :
  9 * (BC / 10) * (h / 10) / 2 = 10 * (BC / 2) * A / 19 :=
sorry

end triangle_area_calculation_l245_24563


namespace minimum_numbers_to_form_triangle_l245_24591

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem minimum_numbers_to_form_triangle :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 1001) →
    16 ≤ S.card →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ {a, b, c} ⊆ S ∧ is_triangle a b c :=
by
  sorry

end minimum_numbers_to_form_triangle_l245_24591


namespace sufficient_but_not_necessary_condition_l245_24569

variables (x y : ℝ)

theorem sufficient_but_not_necessary_condition :
  ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → ((x - 1) * (y - 2) = 0) ∧ (¬ ((x - 1) * (y-2) = 0 → (x - 1)^2 + (y - 2)^2 = 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l245_24569


namespace function_even_iff_a_eq_one_l245_24538

theorem function_even_iff_a_eq_one (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = a * (3^x) + 1/(3^x)) → 
  (∀ x : ℝ, f x = f (-x)) ↔ a = 1 :=
by
  sorry

end function_even_iff_a_eq_one_l245_24538


namespace factorize_polynomial_l245_24578

theorem factorize_polynomial (a x : ℝ) : 
  (x^3 - 3*x^2 + (a + 2)*x - 2*a) = (x^2 - x + a)*(x - 2) :=
by
  sorry

end factorize_polynomial_l245_24578


namespace age_difference_l245_24551

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 18) : C = A - 18 :=
sorry

end age_difference_l245_24551


namespace no_difference_410_l245_24516

theorem no_difference_410 (n : ℕ) (R L a : ℕ) (h1 : R + L = 300)
  (h2 : L = 300 - R)
  (h3 : a ≤ 2 * R)
  (h4 : n = L + a)  :
  ¬ (n = 410) :=
by
  sorry

end no_difference_410_l245_24516


namespace shared_earnings_eq_27_l245_24560

theorem shared_earnings_eq_27
    (shoes_pairs : ℤ) (shoes_cost : ℤ) (shirts : ℤ) (shirts_cost : ℤ)
    (h1 : shoes_pairs = 6) (h2 : shoes_cost = 3)
    (h3 : shirts = 18) (h4 : shirts_cost = 2) :
    (shoes_pairs * shoes_cost + shirts * shirts_cost) / 2 = 27 := by
  sorry

end shared_earnings_eq_27_l245_24560


namespace sum_of_digits_of_9ab_l245_24589

noncomputable def a : ℕ := 10^2023 - 1
noncomputable def b : ℕ := 2*(10^2023 - 1) / 3

def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_9ab :
  digitSum (9 * a * b) = 20235 :=
by
  sorry

end sum_of_digits_of_9ab_l245_24589


namespace bob_expected_difference_l245_24527

-- Required definitions and conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def probability_of_event_s : ℚ := 4 / 7
def probability_of_event_u : ℚ := 2 / 7
def probability_of_event_s_and_u : ℚ := 1 / 7
def number_of_days : ℕ := 365

noncomputable def expected_days_sweetened : ℚ :=
   (probability_of_event_s - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_days_unsweetened : ℚ :=
   (probability_of_event_u - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_difference : ℚ :=
   expected_days_sweetened - expected_days_unsweetened

theorem bob_expected_difference : expected_difference = 135.45 := sorry

end bob_expected_difference_l245_24527


namespace root_conditions_l245_24546

theorem root_conditions (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1^2 - 5 * x1| = a ∧ |x2^2 - 5 * x2| = a) ↔ (a = 0 ∨ a > 25 / 4) := 
by 
  sorry

end root_conditions_l245_24546


namespace sum_of_a_and_b_l245_24515

theorem sum_of_a_and_b (a b : ℝ) (h_neq : a ≠ b) (h_a : a * (a - 4) = 21) (h_b : b * (b - 4) = 21) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l245_24515


namespace primes_div_order_l245_24512

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l245_24512


namespace additional_time_to_empty_tank_l245_24565

-- Definitions based on conditions
def tankCapacity : ℕ := 3200  -- litres
def outletTimeAlone : ℕ := 5  -- hours
def inletRate : ℕ := 4  -- litres/min

-- Calculate rates
def outletRate : ℕ := tankCapacity / outletTimeAlone  -- litres/hour
def inletRatePerHour : ℕ := inletRate * 60  -- Convert litres/min to litres/hour

-- Calculate effective_rate when both pipes open
def effectiveRate : ℕ := outletRate - inletRatePerHour  -- litres/hour

-- Calculate times
def timeWithInletOpen : ℕ := tankCapacity / effectiveRate  -- hours
def additionalTime : ℕ := timeWithInletOpen - outletTimeAlone  -- hours

-- Proof statement
theorem additional_time_to_empty_tank : additionalTime = 3 := by
  -- It's clear from calculation above, we just add sorry for now to skip the proof
  sorry

end additional_time_to_empty_tank_l245_24565


namespace min_rectangles_needed_l245_24548

theorem min_rectangles_needed : ∀ (n : ℕ), n = 12 → (n * n) / (3 * 2) = 24 :=
by sorry

end min_rectangles_needed_l245_24548


namespace ratio_of_third_to_second_building_l245_24555

/-
The tallest building in the world is 100 feet tall. The second tallest is half that tall, the third tallest is some 
fraction of the second tallest building's height, and the fourth is one-fifth as tall as the third. All 4 buildings 
put together are 180 feet tall. What is the ratio of the height of the third tallest building to the second tallest building?

Given H1 = 100, H2 = (1 / 2) * H1, H4 = (1 / 5) * H3, 
and H1 + H2 + H3 + H4 = 180, prove that H3 / H2 = 1 / 2.
-/

theorem ratio_of_third_to_second_building :
  ∀ (H1 H2 H3 H4 : ℝ),
  H1 = 100 →
  H2 = (1 / 2) * H1 →
  H4 = (1 / 5) * H3 →
  H1 + H2 + H3 + H4 = 180 →
  (H3 / H2) = (1 / 2) :=
by
  intros H1 H2 H3 H4 h1_eq h2_half_h1 h4_fifth_h3 total_eq
  /- proof steps go here -/
  sorry

end ratio_of_third_to_second_building_l245_24555


namespace distance_between_cities_l245_24503

theorem distance_between_cities (x : ℝ) (h1 : x ≥ 100) (t : ℝ)
  (A_speed : ℝ := 12) (B_speed : ℝ := 0.05 * x)
  (condition_A : 7 + A_speed * t + B_speed * t = x)
  (condition_B : t = (x - 7) / (A_speed + B_speed)) :
  x = 140 :=
sorry

end distance_between_cities_l245_24503


namespace ted_alex_age_ratio_l245_24596

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio_l245_24596


namespace geometric_sequence_first_term_l245_24567

theorem geometric_sequence_first_term (a r : ℝ)
    (h1 : a * r^2 = 3)
    (h2 : a * r^4 = 27) :
    a = 1 / 3 := by
    sorry

end geometric_sequence_first_term_l245_24567


namespace finitely_many_n_divisors_in_A_l245_24572

-- Lean 4 statement
theorem finitely_many_n_divisors_in_A (A : Finset ℕ) (a : ℕ) (hA : ∀ p ∈ A, Nat.Prime p) (ha : a ≥ 2) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ∃ p : ℕ, p ∣ a^n - 1 ∧ p ∉ A := by
  sorry

end finitely_many_n_divisors_in_A_l245_24572


namespace davi_minimum_spending_l245_24594

-- Define the cost of a single bottle
def singleBottleCost : ℝ := 2.80

-- Define the cost of a box of six bottles
def boxCost : ℝ := 15.00

-- Define the number of bottles Davi needs to buy
def totalBottles : ℕ := 22

-- Calculate the minimum amount Davi will spend
def minimumCost : ℝ := 45.00 + 11.20 

-- The theorem to prove
theorem davi_minimum_spending :
  ∃ minCost : ℝ, minCost = 56.20 ∧ minCost = 3 * boxCost + 4 * singleBottleCost := 
by
  use 56.20
  sorry

end davi_minimum_spending_l245_24594


namespace kimberly_skittles_proof_l245_24564

variable (SkittlesInitial : ℕ) (SkittlesBought : ℕ) (OrangesBought : ℕ)

/-- Kimberly's initial number of Skittles --/
def kimberly_initial_skittles := SkittlesInitial

/-- Skittles Kimberly buys --/
def kimberly_skittles_bought := SkittlesBought

/-- Oranges Kimbery buys (irrelevant for Skittles count) --/
def kimberly_oranges_bought := OrangesBought

/-- Total Skittles Kimberly has --/
def kimberly_total_skittles (SkittlesInitial SkittlesBought : ℕ) : ℕ :=
  SkittlesInitial + SkittlesBought

/-- Proof statement --/
theorem kimberly_skittles_proof (h1 : SkittlesInitial = 5) (h2 : SkittlesBought = 7) : 
  kimberly_total_skittles SkittlesInitial SkittlesBought = 12 :=
by
  rw [h1, h2]
  exact rfl

end kimberly_skittles_proof_l245_24564


namespace minimum_workers_required_l245_24518

theorem minimum_workers_required (total_days : ℕ) (days_elapsed : ℕ) (initial_workers : ℕ) (job_fraction_done : ℚ)
  (remaining_work_fraction : job_fraction_done < 1) 
  (worker_productivity_constant : Prop) : 
  total_days = 40 → days_elapsed = 10 → initial_workers = 10 → job_fraction_done = (1/4) →
  (total_days - days_elapsed) * initial_workers * job_fraction_done = (1 - job_fraction_done) →
  job_fraction_done = 1 → initial_workers = 10 :=
by
  intros;
  sorry

end minimum_workers_required_l245_24518


namespace perimeter_of_polygon_l245_24598

-- Define the dimensions of the strips and their arrangement
def strip_width : ℕ := 4
def strip_length : ℕ := 16
def num_vertical_strips : ℕ := 2
def num_horizontal_strips : ℕ := 2

-- State the problem condition and the expected perimeter
theorem perimeter_of_polygon : 
  let vertical_perimeter := num_vertical_strips * strip_length
  let horizontal_perimeter := num_horizontal_strips * strip_length
  let corner_segments_perimeter := (num_vertical_strips + num_horizontal_strips) * strip_width
  vertical_perimeter + horizontal_perimeter + corner_segments_perimeter = 80 :=
by
  sorry

end perimeter_of_polygon_l245_24598


namespace annie_job_time_l245_24506

noncomputable def annie_time : ℝ :=
  let dan_time := 15
  let dan_rate := 1 / dan_time
  let dan_hours := 6
  let fraction_done_by_dan := dan_rate * dan_hours
  let fraction_left_for_annie := 1 - fraction_done_by_dan
  let annie_work_remaining := fraction_left_for_annie
  let annie_hours := 6
  let annie_rate := annie_work_remaining / annie_hours
  let annie_time := 1 / annie_rate 
  annie_time

theorem annie_job_time :
  annie_time = 3.6 := 
sorry

end annie_job_time_l245_24506


namespace total_onions_l245_24537

theorem total_onions (S SA F J : ℕ) (h1 : S = 4) (h2 : SA = 5) (h3 : F = 9) (h4 : J = 7) : S + SA + F + J = 25 :=
by {
  sorry
}

end total_onions_l245_24537


namespace craig_apples_total_l245_24524

-- Conditions
def initial_apples := 20.0
def additional_apples := 7.0

-- Question turned into a proof problem
theorem craig_apples_total : initial_apples + additional_apples = 27.0 :=
by
  sorry

end craig_apples_total_l245_24524


namespace graduates_distribution_l245_24528

theorem graduates_distribution (n : ℕ) (k : ℕ)
    (h_n : n = 5) (h_k : k = 3)
    (h_dist : ∀ e : Fin k, ∃ g : Finset (Fin n), g.card ≥ 1) :
    ∃ d : ℕ, d = 150 :=
by
  have h_distribution := 150
  use h_distribution
  sorry

end graduates_distribution_l245_24528


namespace race_time_comparison_l245_24544

noncomputable def townSquare : ℝ := 3 / 4 -- distance of one lap in miles
noncomputable def laps : ℕ := 7 -- number of laps
noncomputable def totalDistance : ℝ := laps * townSquare -- total distance of the race in miles
noncomputable def thisYearTime : ℝ := 42 -- time taken by this year's winner in minutes
noncomputable def lastYearTime : ℝ := 47.25 -- time taken by last year's winner in minutes

noncomputable def thisYearPace : ℝ := thisYearTime / totalDistance -- pace of this year's winner in minutes per mile
noncomputable def lastYearPace : ℝ := lastYearTime / totalDistance -- pace of last year's winner in minutes per mile
noncomputable def timeDifference : ℝ := lastYearPace - thisYearPace -- the difference in pace

theorem race_time_comparison : timeDifference = 1 := by
  sorry

end race_time_comparison_l245_24544


namespace dad_steps_90_l245_24532

/-- 
  Given:
  - When Dad takes 3 steps, Masha takes 5 steps.
  - When Masha takes 3 steps, Yasha takes 5 steps.
  - Masha and Yasha together made a total of 400 steps.

  Prove: 
  The number of steps that Dad took is 90.
-/
theorem dad_steps_90 (total_steps: ℕ) (masha_to_dad_ratio: ℕ) (yasha_to_masha_ratio: ℕ) (steps_masha_yasha: ℕ) (h1: masha_to_dad_ratio = 5) (h2: yasha_to_masha_ratio = 5) (h3: steps_masha_yasha = 400) :
  total_steps = 90 :=
by
  sorry

end dad_steps_90_l245_24532


namespace equal_intercepts_on_both_axes_l245_24571

theorem equal_intercepts_on_both_axes (m : ℝ) :
  (5 - 2 * m ≠ 0) ∧
  (- (5 - 2 * m) / (m^2 - 2 * m - 3) = - (5 - 2 * m) / (2 * m^2 + m - 1)) ↔ m = -2 :=
by sorry

end equal_intercepts_on_both_axes_l245_24571


namespace total_rooms_count_l245_24550

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count_l245_24550


namespace scientific_notation_14000000_l245_24599

theorem scientific_notation_14000000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 14000000 = a * 10 ^ n ∧ a = 1.4 ∧ n = 7 :=
by
  sorry

end scientific_notation_14000000_l245_24599


namespace complex_problem_l245_24530

open Complex

theorem complex_problem
  (α θ β : ℝ)
  (h : exp (i * (α + θ)) + exp (i * (β + θ)) = 1 / 3 + (4 / 9) * i) :
  exp (-i * (α + θ)) + exp (-i * (β + θ)) = 1 / 3 - (4 / 9) * i :=
by
  sorry

end complex_problem_l245_24530


namespace problem_solution_l245_24549

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution_l245_24549


namespace female_sample_count_is_correct_l245_24520

-- Definitions based on the given conditions
def total_students : ℕ := 900
def male_students : ℕ := 500
def sample_size : ℕ := 45
def female_students : ℕ := total_students - male_students
def female_sample_size : ℕ := (female_students * sample_size) / total_students

-- The lean statement to prove
theorem female_sample_count_is_correct : female_sample_size = 20 := 
by 
  -- A placeholder to indicate the proof needs to be filled in
  sorry

end female_sample_count_is_correct_l245_24520


namespace common_ratio_of_arithmetic_sequence_l245_24587

theorem common_ratio_of_arithmetic_sequence (S_odd S_even : ℤ) (q : ℤ) 
  (h1 : S_odd + S_even = -240) (h2 : S_odd - S_even = 80) 
  (h3 : q = S_even / S_odd) : q = 2 := 
  sorry

end common_ratio_of_arithmetic_sequence_l245_24587


namespace machine_produces_480_cans_in_8_hours_l245_24575

def cans_produced_in_interval : ℕ := 30
def interval_duration_minutes : ℕ := 30
def hours_worked : ℕ := 8
def minutes_in_hour : ℕ := 60

theorem machine_produces_480_cans_in_8_hours :
  (hours_worked * (minutes_in_hour / interval_duration_minutes) * cans_produced_in_interval) = 480 := by
  sorry

end machine_produces_480_cans_in_8_hours_l245_24575


namespace determine_a_l245_24510

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M m : ℝ)
  (hM : M = max (a^1) (a^2))
  (hm : m = min (a^1) (a^2))
  (hM_m : M = 2 * m) :
  a = 1/2 ∨ a = 2 := 
by sorry

end determine_a_l245_24510


namespace larger_number_l245_24561

theorem larger_number (a b : ℕ) (h1 : 5 * b = 7 * a) (h2 : b - a = 10) : b = 35 :=
sorry

end larger_number_l245_24561


namespace infinite_series_value_l245_24588

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l245_24588


namespace probability_sum_l245_24570

noncomputable def P : ℕ → ℝ := sorry

theorem probability_sum (n : ℕ) (h : n ≥ 7) :
  P n = (1/6) * (P (n-1) + P (n-2) + P (n-3) + P (n-4) + P (n-5) + P (n-6)) :=
sorry

end probability_sum_l245_24570


namespace journey_possibility_l245_24552

noncomputable def possible_start_cities 
  (routes : List (String × String)) 
  (visited : List String) : List String :=
sorry

theorem journey_possibility :
  possible_start_cities 
    [("Saint Petersburg", "Tver"), 
     ("Yaroslavl", "Nizhny Novgorod"), 
     ("Moscow", "Kazan"), 
     ("Nizhny Novgorod", "Kazan"), 
     ("Moscow", "Tver"), 
     ("Moscow", "Nizhny Novgorod")]
    ["Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan"] 
  = ["Saint Petersburg", "Yaroslavl"] :=
sorry

end journey_possibility_l245_24552


namespace perpendicular_vectors_l245_24502

variable {t : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (ht : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) : t = -5 :=
sorry

end perpendicular_vectors_l245_24502


namespace jay_more_points_than_tobee_l245_24585

-- Declare variables.
variables (x J S : ℕ)

-- Given conditions
def Tobee_points := 4
def Jay_points := Tobee_points + x -- Jay_score is 4 + x
def Sean_points := (Tobee_points + Jay_points) - 2 -- Sean_score is 4 + Jay - 2

-- The total score condition
def total_score_condition := Tobee_points + Jay_points + Sean_points = 26

-- The main statement to be proven
theorem jay_more_points_than_tobee (h : total_score_condition) : J - Tobee_points = 6 :=
sorry

end jay_more_points_than_tobee_l245_24585


namespace equation_of_line_through_P_l245_24576

theorem equation_of_line_through_P (P : (ℝ × ℝ)) (A B : (ℝ × ℝ))
  (hP : P = (1, 3))
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : A.2 = 0)
  (hB : B.1 = 0) :
  ∃ c : ℝ, 3 * c + 1 = 3 ∧ (3 * A.1 / c + A.2 / 6 = 1) ∧ (3 * B.1 / c + B.2 / 6 = 1) := sorry

end equation_of_line_through_P_l245_24576


namespace find_k_common_term_l245_24519

def sequence_a (k : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 
  else if n = 2 then k 
  else if n = 3 then 3*k - 3 
  else if n = 4 then 6*k - 8 
  else (n * (n-1) * (k-2)) / 2 + n

def is_fermat (x : ℕ) : Prop :=
  ∃ m : ℕ, x = 2^(2^m) + 1

theorem find_k_common_term (k : ℕ) :
  k > 2 → ∃ n m : ℕ, sequence_a k n = 2^(2^m) + 1 :=
by
  sorry

end find_k_common_term_l245_24519


namespace wrongly_entered_mark_l245_24554

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ marks_instead_of_45 number_of_pupils (total_avg_increase : ℝ),
     marks_instead_of_45 = 45 ∧
     number_of_pupils = 44 ∧
     total_avg_increase = 0.5 →
     x = marks_instead_of_45 + total_avg_increase * number_of_pupils) →
  x = 67 :=
by
  intro h
  sorry

end wrongly_entered_mark_l245_24554


namespace unique_triplets_l245_24595

theorem unique_triplets (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + 
               |c * x + a * y + b * z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) :=
sorry

end unique_triplets_l245_24595


namespace num_factors_1728_l245_24522

open Nat

noncomputable def num_factors (n : ℕ) : ℕ :=
  (6 + 1) * (3 + 1)

theorem num_factors_1728 : 
  num_factors 1728 = 28 := by
  sorry

end num_factors_1728_l245_24522


namespace side_length_of_largest_square_l245_24536

theorem side_length_of_largest_square (A_cross : ℝ) (s : ℝ)
  (h1 : A_cross = 810) : s = 36 :=
  have h_large_squares : 2 * (s / 2)^2 = s^2 / 2 := by sorry
  have h_small_squares : 2 * (s / 4)^2 = s^2 / 8 := by sorry
  have h_combined_area : s^2 / 2 + s^2 / 8 = 810 := by sorry
  have h_final : 5 * s^2 / 8 = 810 := by sorry
  have h_s2 : s^2 = 1296 := by sorry
  have h_s : s = 36 := by sorry
  h_s

end side_length_of_largest_square_l245_24536


namespace original_perimeter_not_necessarily_multiple_of_four_l245_24597

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four_l245_24597


namespace find_a_l245_24500

def A (x : ℝ) := (x^2 - 4 ≤ 0)
def B (x : ℝ) (a : ℝ) := (2 * x + a ≤ 0)
def C (x : ℝ) := (-2 ≤ x ∧ x ≤ 1)

theorem find_a (a : ℝ) : (∀ x : ℝ, A x → B x a → C x) → a = -2 :=
sorry

end find_a_l245_24500


namespace smallest_four_digit_multiple_of_53_l245_24514

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l245_24514


namespace polynomial_characterization_l245_24507
open Polynomial

noncomputable def satisfies_functional_eq (P : Polynomial ℝ) :=
  ∀ (a b c : ℝ), 
  P.eval (a + b - 2*c) + P.eval (b + c - 2*a) + P.eval (c + a - 2*b) = 
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_characterization (P : Polynomial ℝ) :
  satisfies_functional_eq P ↔ 
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X + Polynomial.C b) ∨
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X) :=
sorry

end polynomial_characterization_l245_24507


namespace quadratic_inequality_condition_l245_24559

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) :=
sorry

end quadratic_inequality_condition_l245_24559


namespace max_roses_l245_24566

theorem max_roses (budget : ℝ) (indiv_price : ℝ) (dozen_1_price : ℝ) (dozen_2_price : ℝ) (dozen_5_price : ℝ) (hundred_price : ℝ) 
  (budget_eq : budget = 1000) (indiv_price_eq : indiv_price = 5.30) (dozen_1_price_eq : dozen_1_price = 36) 
  (dozen_2_price_eq : dozen_2_price = 50) (dozen_5_price_eq : dozen_5_price = 110) (hundred_price_eq : hundred_price = 180) : 
  ∃ max_roses : ℕ, max_roses = 548 :=
by
  sorry

end max_roses_l245_24566


namespace part1_part2_part3_l245_24542

-- Part (1)
theorem part1 (m n : ℤ) (h1 : m - n = -1) : 2 * (m - n)^2 + 18 = 20 := 
sorry

-- Part (2)
theorem part2 (m n : ℤ) (h2 : m^2 + 2 * m * n = 10) (h3 : n^2 + 3 * m * n = 6) : 2 * m^2 + n^2 + 7 * m * n = 26 :=
sorry

-- Part (3)
theorem part3 (a b c m x : ℤ) (h4: ax^5 + bx^3 + cx - 5 = m) (h5: x = -1) : ax^5 + bx^3 + cx - 5 = -m - 10 :=
sorry

end part1_part2_part3_l245_24542


namespace power_mod_2040_l245_24539

theorem power_mod_2040 : (6^2040) % 13 = 1 := by
  -- Skipping the proof as the problem only requires the statement
  sorry

end power_mod_2040_l245_24539


namespace washing_machine_cost_l245_24501

variable (W D : ℝ)
variable (h1 : D = W - 30)
variable (h2 : 0.90 * (W + D) = 153)

theorem washing_machine_cost :
  W = 100 := by
  sorry

end washing_machine_cost_l245_24501


namespace joe_height_is_82_l245_24505

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l245_24505


namespace total_value_of_bills_l245_24592

theorem total_value_of_bills 
  (total_bills : Nat := 12) 
  (num_5_dollar_bills : Nat := 4) 
  (num_10_dollar_bills : Nat := 8)
  (value_5_dollar_bill : Nat := 5)
  (value_10_dollar_bill : Nat := 10) :
  (num_5_dollar_bills * value_5_dollar_bill + num_10_dollar_bills * value_10_dollar_bill = 100) :=
by
  sorry

end total_value_of_bills_l245_24592


namespace max_trains_final_count_l245_24574

-- Define the conditions
def trains_per_birthdays : Nat := 1
def trains_per_christmas : Nat := 2
def trains_per_easter : Nat := 3
def years : Nat := 7

-- Function to calculate total trains after 7 years
def total_trains_after_years (trains_per_years : Nat) (num_years : Nat) : Nat :=
  trains_per_years * num_years

-- Calculate inputs
def trains_per_year : Nat := trains_per_birthdays + trains_per_christmas + trains_per_easter
def total_initial_trains : Nat := total_trains_after_years trains_per_year years

-- Bonus and final steps
def bonus_trains_from_cousins (initial_trains : Nat) : Nat := initial_trains / 2
def final_total_trains (initial_trains : Nat) (bonus_trains : Nat) : Nat :=
  let after_bonus := initial_trains + bonus_trains
  let additional_from_parents := after_bonus * 3
  after_bonus + additional_from_parents

-- Main theorem
theorem max_trains_final_count : final_total_trains total_initial_trains (bonus_trains_from_cousins total_initial_trains) = 252 := by
  sorry

end max_trains_final_count_l245_24574


namespace ratio_eq_one_l245_24582

theorem ratio_eq_one (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
sorry

end ratio_eq_one_l245_24582


namespace find_S3_l245_24540

noncomputable def geometric_sum (n : ℕ) : ℕ := sorry  -- Placeholder for the sum function.

theorem find_S3 (S : ℕ → ℕ) (hS6 : S 6 = 30) (hS9 : S 9 = 70) : S 3 = 10 :=
by
  -- Establish the needed conditions and equation 
  have h : (S 6 - S 3) ^ 2 = (S 9 - S 6) * S 3 := sorry
  -- Substitute given S6 and S9 into the equation and solve
  exact sorry

end find_S3_l245_24540


namespace find_interval_l245_24579

theorem find_interval (x : ℝ) : (x > 3/4 ∧ x < 4/5) ↔ (5 * x + 1 > 3 ∧ 5 * x + 1 < 5 ∧ 4 * x > 3 ∧ 4 * x < 5) :=
by
  sorry

end find_interval_l245_24579


namespace yellow_ball_count_l245_24547

def total_balls : ℕ := 500
def red_balls : ℕ := total_balls / 3
def remaining_after_red : ℕ := total_balls - red_balls
def blue_balls : ℕ := remaining_after_red / 5
def remaining_after_blue : ℕ := remaining_after_red - blue_balls
def green_balls : ℕ := remaining_after_blue / 4
def yellow_balls : ℕ := total_balls - (red_balls + blue_balls + green_balls)

theorem yellow_ball_count : yellow_balls = 201 := by
  sorry

end yellow_ball_count_l245_24547


namespace shopkeeper_loss_percentages_l245_24521

theorem shopkeeper_loss_percentages 
  (TypeA : Type) (TypeB : Type) (TypeC : Type)
  (theft_percentage_A : ℝ) (theft_percentage_B : ℝ) (theft_percentage_C : ℝ)
  (hA : theft_percentage_A = 0.20)
  (hB : theft_percentage_B = 0.25)
  (hC : theft_percentage_C = 0.30)
  :
  (theft_percentage_A = 0.20 ∧ theft_percentage_B = 0.25 ∧ theft_percentage_C = 0.30) ∧
  ((theft_percentage_A + theft_percentage_B + theft_percentage_C) / 3 = 0.25) :=
by
  sorry

end shopkeeper_loss_percentages_l245_24521


namespace trigonometric_identity_l245_24586

open Real

-- Lean 4 statement
theorem trigonometric_identity (α β γ x : ℝ) :
  (sin (x - β) * sin (x - γ) / (sin (α - β) * sin (α - γ))) +
  (sin (x - γ) * sin (x - α) / (sin (β - γ) * sin (β - α))) +
  (sin (x - α) * sin (x - β) / (sin (γ - α) * sin (γ - β))) = 1 := 
sorry

end trigonometric_identity_l245_24586


namespace billy_soda_distribution_l245_24573

theorem billy_soda_distribution (sisters : ℕ) (brothers : ℕ) (total_sodas : ℕ) (total_siblings : ℕ)
  (h1 : total_sodas = 12)
  (h2 : sisters = 2)
  (h3 : brothers = 2 * sisters)
  (h4 : total_siblings = sisters + brothers) :
  total_sodas / total_siblings = 2 :=
by
  sorry

end billy_soda_distribution_l245_24573


namespace inequality_holds_for_all_l245_24577

theorem inequality_holds_for_all (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ α β : ℝ, ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) → m = n :=
by sorry

end inequality_holds_for_all_l245_24577


namespace find_k_l245_24593

theorem find_k (k : ℝ) (x : ℝ) :
  x^2 + k * x + 1 = 0 ∧ x^2 - x - k = 0 → k = 2 := 
sorry

end find_k_l245_24593


namespace smallest_integer_k_l245_24509

theorem smallest_integer_k : ∀ (k : ℕ), (64^k > 4^16) → k ≥ 6 :=
by
  sorry

end smallest_integer_k_l245_24509


namespace Marla_laps_per_hour_l245_24590

theorem Marla_laps_per_hour (M : ℝ) :
  (0.8 * M = 0.8 * 5 + 4) → M = 10 :=
by
  sorry

end Marla_laps_per_hour_l245_24590


namespace find_first_term_geom_seq_l245_24543

noncomputable def first_term (a r : ℝ) := a

theorem find_first_term_geom_seq 
  (a r : ℝ) 
  (h1 : a * r ^ 3 = 720) 
  (h2 : a * r ^ 6 = 5040) : 
  first_term a r = 720 / 7 := 
sorry

end find_first_term_geom_seq_l245_24543


namespace at_least_three_bushes_with_same_number_of_flowers_l245_24556

-- Defining the problem using conditions as definitions.
theorem at_least_three_bushes_with_same_number_of_flowers (n : ℕ) (f : Fin n → ℕ) (h1 : n = 201)
  (h2 : ∀ (i : Fin n), 1 ≤ f i ∧ f i ≤ 100) : 
  ∃ (x : ℕ), (∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ f i1 = x ∧ f i2 = x ∧ f i3 = x) := 
by
  sorry

end at_least_three_bushes_with_same_number_of_flowers_l245_24556


namespace sum_of_first_8_terms_l245_24583

theorem sum_of_first_8_terms (a : ℝ) (h : 15 * a = 1) : 
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a + 128 * a) = 17 :=
by
  sorry

end sum_of_first_8_terms_l245_24583


namespace simplify_expression_l245_24534

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l245_24534


namespace smallest_x_such_that_sum_is_cubic_l245_24526

/-- 
  Given a positive integer x, the sum of the sequence x, x+3, x+6, x+9, and x+12 should be a perfect cube.
  Prove that the smallest such x is 19.
-/
theorem smallest_x_such_that_sum_is_cubic : 
  ∃ (x : ℕ), 0 < x ∧ (∃ k : ℕ, 5 * x + 30 = k^3) ∧ ∀ y : ℕ, 0 < y → (∃ m : ℕ, 5 * y + 30 = m^3) → y ≥ x :=
sorry

end smallest_x_such_that_sum_is_cubic_l245_24526


namespace albert_earnings_l245_24558

theorem albert_earnings (E E_final : ℝ) : 
  (0.90 * (E * 1.14) = 678) → 
  (E_final = 0.90 * (E * 1.15 * 1.20)) → 
  E_final = 819.72 :=
by
  sorry

end albert_earnings_l245_24558


namespace negation_of_universal_proposition_l245_24504

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 2^x - 1 > 0)) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l245_24504


namespace molecular_weight_correct_l245_24525

-- Definition of atomic weights for the elements
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Number of atoms in Ascorbic acid (C6H8O6)
def count_C : ℕ := 6
def count_H : ℕ := 8
def count_O : ℕ := 6

-- Calculation of molecular weight
def molecular_weight_ascorbic_acid : ℝ :=
  (count_C * atomic_weight_C) +
  (count_H * atomic_weight_H) +
  (count_O * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_ascorbic_acid = 176.124 :=
by sorry


end molecular_weight_correct_l245_24525


namespace victor_percentage_of_marks_l245_24508

theorem victor_percentage_of_marks (marks_obtained max_marks : ℝ) (percentage : ℝ) 
  (h_marks_obtained : marks_obtained = 368) 
  (h_max_marks : max_marks = 400) 
  (h_percentage : percentage = (marks_obtained / max_marks) * 100) : 
  percentage = 92 := by
sorry

end victor_percentage_of_marks_l245_24508


namespace quadratic_expression_always_positive_l245_24531

theorem quadratic_expression_always_positive (x y : ℝ) : 
  x^2 - 4 * x * y + 6 * y^2 - 4 * y + 3 > 0 :=
by 
  sorry

end quadratic_expression_always_positive_l245_24531
