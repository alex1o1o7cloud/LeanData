import Mathlib

namespace abs_eq_solution_l575_57507

theorem abs_eq_solution (x : ℚ) : |x - 2| = |x + 3| → x = -1 / 2 :=
by
  sorry

end abs_eq_solution_l575_57507


namespace tangent_line_to_parabola_l575_57514

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 16 * x) →
  (28 ^ 2 - 4 * 1 * (4 * k) = 0) → k = 49 :=
by
  intro h
  intro h_discriminant
  have discriminant_eq_zero : 28 ^ 2 - 4 * 1 * (4 * k) = 0 := h_discriminant
  sorry

end tangent_line_to_parabola_l575_57514


namespace combined_time_third_attempt_l575_57552

noncomputable def first_lock_initial : ℕ := 5
noncomputable def second_lock_initial : ℕ := 3 * first_lock_initial - 3
noncomputable def combined_initial : ℕ := 5 * second_lock_initial

noncomputable def first_lock_second_attempt : ℝ := first_lock_initial - 0.1 * first_lock_initial
noncomputable def first_lock_third_attempt : ℝ := first_lock_second_attempt - 0.1 * first_lock_second_attempt

noncomputable def second_lock_second_attempt : ℝ := second_lock_initial - 0.15 * second_lock_initial
noncomputable def second_lock_third_attempt : ℝ := second_lock_second_attempt - 0.15 * second_lock_second_attempt

noncomputable def combined_third_attempt : ℝ := 5 * second_lock_third_attempt

theorem combined_time_third_attempt : combined_third_attempt = 43.35 :=
by
  sorry

end combined_time_third_attempt_l575_57552


namespace expand_product_l575_57518

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  -- No proof required, just state the theorem
  sorry

end expand_product_l575_57518


namespace laborer_monthly_income_l575_57510

variable (I : ℝ)

noncomputable def average_expenditure_six_months := 70 * 6
noncomputable def debt_condition := I * 6 < average_expenditure_six_months
noncomputable def expenditure_next_four_months := 60 * 4
noncomputable def total_income_next_four_months := expenditure_next_four_months + (average_expenditure_six_months - I * 6) + 30

theorem laborer_monthly_income (h1 : debt_condition I) (h2 : total_income_next_four_months I = I * 4) :
  I = 69 :=
by
  sorry

end laborer_monthly_income_l575_57510


namespace inequality_convex_l575_57578

theorem inequality_convex (x y a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : a + b = 1) : 
  (a * x + b * y) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 := 
sorry

end inequality_convex_l575_57578


namespace sum_of_coordinates_l575_57560

def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g x)^3

theorem sum_of_coordinates (hg : g 4 = 8) : 4 + h 4 = 516 :=
by
  sorry

end sum_of_coordinates_l575_57560


namespace seating_arrangements_count_l575_57513

-- Define the main entities: the three teams and the conditions
inductive Person
| Jupitarian
| Saturnian
| Neptunian

open Person

-- Define the seating problem constraints
def valid_arrangement (seating : Fin 12 → Person) : Prop :=
  seating 0 = Jupitarian ∧ seating 11 = Neptunian ∧
  (∀ i, seating (i % 12) = Jupitarian → seating ((i + 11) % 12) ≠ Neptunian) ∧
  (∀ i, seating (i % 12) = Neptunian → seating ((i + 11) % 12) ≠ Saturnian) ∧
  (∀ i, seating (i % 12) = Saturnian → seating ((i + 11) % 12) ≠ Jupitarian)

-- Main theorem: The number of valid arrangements is 225 * (4!)^3
theorem seating_arrangements_count :
  ∃ M : ℕ, (M = 225) ∧ ∃ arrangements : Fin 12 → Person, valid_arrangement arrangements :=
sorry

end seating_arrangements_count_l575_57513


namespace functional_relationship_find_selling_price_maximum_profit_l575_57596

noncomputable def linear_relation (x : ℤ) : ℤ := -5 * x + 150
def profit_function (x : ℤ) : ℤ := -5 * x * x + 200 * x - 1500

theorem functional_relationship (x : ℤ) (hx : 10 ≤ x ∧ x ≤ 15) : linear_relation x = -5 * x + 150 :=
by sorry

theorem find_selling_price (h : ∃ x : ℤ, (10 ≤ x ∧ x ≤ 15) ∧ ((-5 * x + 150) * (x - 10) = 320)) :
  ∃ x : ℤ, x = 14 :=
by sorry

theorem maximum_profit (hx : 10 ≤ 15 ∧ 15 ≤ 15) : profit_function 15 = 375 :=
by sorry

end functional_relationship_find_selling_price_maximum_profit_l575_57596


namespace portraits_after_lunch_before_gym_class_l575_57534

-- Define the total number of students in the class
def total_students : ℕ := 24

-- Define the number of students who had their portraits taken before lunch
def students_before_lunch : ℕ := total_students / 3

-- Define the number of students who have not yet had their picture taken after gym class
def students_after_gym_class : ℕ := 6

-- Define the number of students who had their portraits taken before gym class
def students_before_gym_class : ℕ := total_students - students_after_gym_class

-- Define the number of students who had their portraits taken after lunch but before gym class
def students_after_lunch_before_gym_class : ℕ := students_before_gym_class - students_before_lunch

-- Statement of the theorem
theorem portraits_after_lunch_before_gym_class :
  students_after_lunch_before_gym_class = 10 :=
by
  -- The proof is omitted
  sorry

end portraits_after_lunch_before_gym_class_l575_57534


namespace a_2n_is_perfect_square_l575_57544

-- Define the sequence a_n as per the problem's conditions
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n - 1) + a (n - 3) + a (n - 4)

-- Define the Fibonacci sequence for comparison
def fib (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

-- Key theorem to prove: a_{2n} is a perfect square
theorem a_2n_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, a (2 * n) = k * k :=
sorry

end a_2n_is_perfect_square_l575_57544


namespace gears_together_again_l575_57588

theorem gears_together_again (r₁ r₂ : ℕ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) : 
  (∃ t : ℕ, t = Nat.lcm r₁ r₂ / r₁ ∨ t = Nat.lcm r₁ r₂ / r₂) → 5 = Nat.lcm r₁ r₂ / min r₁ r₂ := 
by
  sorry

end gears_together_again_l575_57588


namespace total_shaded_area_of_rectangles_l575_57574

theorem total_shaded_area_of_rectangles (w1 l1 w2 l2 ow ol : ℕ) 
  (h1 : w1 = 4) (h2 : l1 = 12) (h3 : w2 = 5) (h4 : l2 = 10) (h5 : ow = 4) (h6 : ol = 5) :
  (w1 * l1 + w2 * l2 - ow * ol = 78) :=
by
  sorry

end total_shaded_area_of_rectangles_l575_57574


namespace vector_sum_correct_l575_57522

-- Define the three vectors
def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 6)
def v3 : ℝ × ℝ := (2, -8)

-- Define the expected result
def expected_sum : ℝ × ℝ := (3, -5)

-- Define vector addition (component-wise)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- The theorem statement
theorem vector_sum_correct : vector_add (vector_add v1 v2) v3 = expected_sum := by
  sorry

end vector_sum_correct_l575_57522


namespace meeting_equation_correct_l575_57595

-- Define the conditions
def distance : ℝ := 25
def time : ℝ := 3
def speed_Xiaoming : ℝ := 4
def speed_Xiaogang (x : ℝ) : ℝ := x

-- The target equation derived from conditions which we need to prove valid.
theorem meeting_equation_correct (x : ℝ) : 3 * (speed_Xiaoming + speed_Xiaogang x) = distance :=
by
  sorry

end meeting_equation_correct_l575_57595


namespace inscribed_rectangle_area_l575_57535

theorem inscribed_rectangle_area (h a b x : ℝ) (ha_gt_b : a > b) :
  ∃ A : ℝ, A = (b * x / h) * (h - x) :=
by
  sorry

end inscribed_rectangle_area_l575_57535


namespace non_similar_triangles_with_arithmetic_angles_l575_57582

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles_l575_57582


namespace angle_C_measure_l575_57583

theorem angle_C_measure 
  (p q : Prop) 
  (h1 : p) (h2 : q) 
  (A B C : ℝ) 
  (h_parallel : p = q) 
  (h_A_B : A = B / 10) 
  (h_straight_line : B + C = 180) 
  : C = 16.36 := 
sorry

end angle_C_measure_l575_57583


namespace gcd_153_119_l575_57511

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l575_57511


namespace total_degree_difference_l575_57504

-- Definitions based on conditions
def timeStart : ℕ := 12 * 60  -- noon in minutes
def timeEnd : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes
def numTimeZones : ℕ := 3  -- Three time zones
def degreesInCircle : ℕ := 360  -- Degrees in a full circle

-- Calculate degrees moved by each hand
def degreesMovedByHourHand : ℚ := (timeEnd - timeStart) / (12 * 60) * degreesInCircle
def degreesMovedByMinuteHand : ℚ := (timeEnd - timeStart) % 60 * (degreesInCircle / 60)
def degreesMovedBySecondHand : ℕ := 0  -- At 2:30 PM, second hand is at initial position

-- Calculate total degree difference for all three hands and time zones
def totalDegrees : ℚ := 
  (degreesMovedByHourHand + degreesMovedByMinuteHand + degreesMovedBySecondHand) * numTimeZones

-- Theorem statement to prove
theorem total_degree_difference :
  totalDegrees = 765 := by
  sorry

end total_degree_difference_l575_57504


namespace student_total_marks_l575_57524

theorem student_total_marks (total_questions correct_answers incorrect_mark correct_mark : ℕ) 
                             (H1 : total_questions = 60) 
                             (H2 : correct_answers = 34)
                             (H3 : incorrect_mark = 1)
                             (H4 : correct_mark = 4) :
  ((correct_answers * correct_mark) - ((total_questions - correct_answers) * incorrect_mark)) = 110 := 
by {
  -- The proof goes here.
  sorry
}

end student_total_marks_l575_57524


namespace weekly_milk_production_l575_57541

-- Define the conditions
def num_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the proof that total weekly milk production is 1820 liters
theorem weekly_milk_production : num_cows * milk_per_cow_per_day * days_per_week = 1820 := by
  sorry

end weekly_milk_production_l575_57541


namespace area_outside_smaller_squares_l575_57536

theorem area_outside_smaller_squares (side_large : ℕ) (side_small1 : ℕ) (side_small2 : ℕ)
  (no_overlap : Prop) (side_large_eq : side_large = 9)
  (side_small1_eq : side_small1 = 4)
  (side_small2_eq : side_small2 = 2) :
  (side_large * side_large - (side_small1 * side_small1 + side_small2 * side_small2)) = 61 :=
by
  sorry

end area_outside_smaller_squares_l575_57536


namespace present_population_l575_57591

theorem present_population (P : ℕ) (h1 : P * 11 / 10 = 264) : P = 240 :=
by sorry

end present_population_l575_57591


namespace sum_angles_acute_l575_57521

open Real

theorem sum_angles_acute (A B C : ℝ) (hA_ac : A < π / 2) (hB_ac : B < π / 2) (hC_ac : C < π / 2)
  (h_angle_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end sum_angles_acute_l575_57521


namespace parallel_planes_mn_l575_57561

theorem parallel_planes_mn (m n : ℝ) (a b : ℝ × ℝ × ℝ) (α β : Type) (h1 : a = (0, 1, m)) (h2 : b = (0, n, -3)) 
  (h3 : ∃ k : ℝ, a = (k • b)) : m * n = -3 :=
by
  -- Proof would be here
  sorry

end parallel_planes_mn_l575_57561


namespace no_such_nat_n_l575_57529

theorem no_such_nat_n :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → (10 * (10 * a + n) + b) % (10 * a + b) = 0 :=
by
  sorry

end no_such_nat_n_l575_57529


namespace math_problem_l575_57562

noncomputable def a : ℝ := 0.137
noncomputable def b : ℝ := 0.098
noncomputable def c : ℝ := 0.123
noncomputable def d : ℝ := 0.086

theorem math_problem : 
  ( ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) ) = 4.6886 := 
  sorry

end math_problem_l575_57562


namespace greatest_xy_value_l575_57543

theorem greatest_xy_value (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 7 * x + 4 * y = 140) :
  (∀ z : ℕ, (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ z = x * y) → z ≤ 168) ∧
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ 168 = x * y) :=
sorry

end greatest_xy_value_l575_57543


namespace drawing_specific_cards_from_two_decks_l575_57584

def prob_of_drawing_specific_cards (total_cards_deck1 total_cards_deck2 : ℕ) 
  (specific_card1 specific_card2 : ℕ) : ℚ :=
(specific_card1 / total_cards_deck1) * (specific_card2 / total_cards_deck2)

theorem drawing_specific_cards_from_two_decks :
  prob_of_drawing_specific_cards 52 52 1 1 = 1 / 2704 :=
by
  -- The proof can be filled in here
  sorry

end drawing_specific_cards_from_two_decks_l575_57584


namespace find_length_of_street_l575_57548

-- Definitions based on conditions
def area_street (L : ℝ) : ℝ := L^2
def area_forest (L : ℝ) : ℝ := 3 * (area_street L)
def num_trees (L : ℝ) : ℝ := 4 * (area_forest L)

-- Statement to prove
theorem find_length_of_street (L : ℝ) (h : num_trees L = 120000) : L = 100 := by
  sorry

end find_length_of_street_l575_57548


namespace determine_digits_l575_57590

def product_consecutive_eq_120_times_ABABAB (n A B : ℕ) : Prop :=
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 * (A * 101010101 + B * 10101010 + A * 1010101 + B * 101010 + A * 10101 + B * 1010 + A * 101 + B * 10 + A)

theorem determine_digits (A B : ℕ) (h : ∃ n, product_consecutive_eq_120_times_ABABAB n A B):
  A = 5 ∧ B = 7 :=
sorry

end determine_digits_l575_57590


namespace Jill_tax_on_clothing_l575_57538

theorem Jill_tax_on_clothing 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ) (total_spent : ℝ) (tax_clothing : ℝ) 
  (tax_other_rate : ℝ) (total_tax_rate : ℝ) 
  (h_clothing : spent_clothing = 0.5 * total_spent) 
  (h_food : spent_food = 0.2 * total_spent) 
  (h_other : spent_other = 0.3 * total_spent) 
  (h_other_tax : tax_other_rate = 0.1) 
  (h_total_tax : total_tax_rate = 0.055) 
  (h_total_spent : total_spent = 100):
  (tax_clothing * spent_clothing + tax_other_rate * spent_other) = total_tax_rate * total_spent → 
  tax_clothing = 0.05 :=
by
  sorry

end Jill_tax_on_clothing_l575_57538


namespace quadratic_inequality_solution_l575_57530

theorem quadratic_inequality_solution (m : ℝ) (h : m ≠ 0) : 
  (∃ x : ℝ, m * x^2 - x + 1 < 0) ↔ (m ∈ Set.Iio 0 ∨ m ∈ Set.Ioo 0 (1 / 4)) :=
by
  sorry

end quadratic_inequality_solution_l575_57530


namespace rotated_ellipse_sum_is_four_l575_57549

noncomputable def rotated_ellipse_center (h' k' : ℝ) : Prop :=
h' = 3 ∧ k' = -5

noncomputable def rotated_ellipse_axes (a' b' : ℝ) : Prop :=
a' = 4 ∧ b' = 2

noncomputable def rotated_ellipse_sum (h' k' a' b' : ℝ) : ℝ :=
h' + k' + a' + b'

theorem rotated_ellipse_sum_is_four (h' k' a' b' : ℝ) 
  (hc : rotated_ellipse_center h' k') (ha : rotated_ellipse_axes a' b') :
  rotated_ellipse_sum h' k' a' b' = 4 :=
by
  -- The proof would be provided here.
  -- Since we're asked not to provide the proof but just to ensure the statement is correct, we use sorry.
  sorry

end rotated_ellipse_sum_is_four_l575_57549


namespace cars_meet_time_l575_57517

theorem cars_meet_time 
  (L : ℕ) (v1 v2 : ℕ) (t : ℕ)
  (H1 : L = 333)
  (H2 : v1 = 54)
  (H3 : v2 = 57)
  (H4 : v1 * t + v2 * t = L) : 
  t = 3 :=
by
  -- Insert proof here
  sorry

end cars_meet_time_l575_57517


namespace a_minus_b_range_l575_57516

noncomputable def range_of_a_minus_b (a b : ℝ) : Set ℝ :=
  {x | -2 < a ∧ a < 1 ∧ 0 < b ∧ b < 4 ∧ x = a - b}

theorem a_minus_b_range (a b : ℝ) (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) :
  ∃ x, range_of_a_minus_b a b x ∧ (-6 < x ∧ x < 1) :=
by
  sorry

end a_minus_b_range_l575_57516


namespace lcm_12_15_18_l575_57528

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l575_57528


namespace exist_two_pies_differing_in_both_l575_57577

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l575_57577


namespace a_can_finish_remaining_work_in_5_days_l575_57569

theorem a_can_finish_remaining_work_in_5_days (a_work_rate b_work_rate : ℝ) (total_days_b_works : ℝ):
  a_work_rate = 1/15 → 
  b_work_rate = 1/15 → 
  total_days_b_works = 10 → 
  ∃ (remaining_days_for_a : ℝ), remaining_days_for_a = 5 :=
by
  intros h1 h2 h3
  -- We are skipping the proof itself
  sorry

end a_can_finish_remaining_work_in_5_days_l575_57569


namespace unique_pair_odd_prime_l575_57581

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end unique_pair_odd_prime_l575_57581


namespace max_truthful_gnomes_l575_57505

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l575_57505


namespace A_plus_B_l575_57570

theorem A_plus_B {A B : ℚ} (h : ∀ x : ℚ, (Bx - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) : 
  A + B = 33 / 5 := sorry

end A_plus_B_l575_57570


namespace problem_statement_l575_57501

theorem problem_statement (c d : ℤ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 :=
by
  sorry

end problem_statement_l575_57501


namespace composite_integer_expression_l575_57519

theorem composite_integer_expression (n : ℕ) (h : n > 1) (hn : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 :=
by
  sorry

end composite_integer_expression_l575_57519


namespace steven_has_72_shirts_l575_57572

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l575_57572


namespace discount_percentage_l575_57509

noncomputable def cost_price : ℝ := 100
noncomputable def profit_with_discount : ℝ := 0.32 * cost_price
noncomputable def profit_without_discount : ℝ := 0.375 * cost_price

noncomputable def sp_with_discount : ℝ := cost_price + profit_with_discount
noncomputable def sp_without_discount : ℝ := cost_price + profit_without_discount

noncomputable def discount_amount : ℝ := sp_without_discount - sp_with_discount
noncomputable def percentage_discount : ℝ := (discount_amount / sp_without_discount) * 100

theorem discount_percentage : percentage_discount = 4 :=
by
  -- proof steps
  sorry

end discount_percentage_l575_57509


namespace brenda_age_l575_57500

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l575_57500


namespace oliver_earning_correct_l575_57598

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l575_57598


namespace max_plates_l575_57550

/-- Bill can buy pans, pots, and plates for 3, 5, and 10 dollars each, respectively.
    What is the maximum number of plates he can purchase if he must buy at least
    two of each item and will spend exactly 100 dollars? -/
theorem max_plates (x y z : ℕ) (hx : x ≥ 2) (hy : y ≥ 2) (hz : z ≥ 2) 
  (h_cost : 3 * x + 5 * y + 10 * z = 100) : z = 8 :=
sorry

end max_plates_l575_57550


namespace gg3_eq_585_over_368_l575_57564

def g (x : ℚ) : ℚ := 2 * x⁻¹ + (2 * x⁻¹) / (1 + 2 * x⁻¹)

theorem gg3_eq_585_over_368 : g (g 3) = 585 / 368 := 
  sorry

end gg3_eq_585_over_368_l575_57564


namespace find_a_solution_set_a_negative_l575_57553

-- Definitions
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Problem 1: Prove the value of 'a'
theorem find_a (h : ∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ -1/2)) :
  a = -2 :=
sorry

-- Problem 2: Prove the solution sets when a < 0
theorem solution_set_a_negative (h : a < 0) :
  (a = -1 → (∀ x : ℝ, quadratic_inequality a x ↔ x = -1)) ∧
  (a < -1 → (∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ 1/a))) ∧
  (-1 < a ∧ a < 0 → (∀ x : ℝ, quadratic_inequality a x ↔ (1/a ≤ x ∧ x ≤ -1))) :=
sorry

end find_a_solution_set_a_negative_l575_57553


namespace probability_drawing_red_l575_57525

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l575_57525


namespace minimize_y_l575_57547

noncomputable def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) ∧ x = (2 * a + 3 * b) / 5 :=
sorry

end minimize_y_l575_57547


namespace shaded_percentage_of_grid_l575_57585

def percent_shaded (total_squares shaded_squares : ℕ) : ℚ :=
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100

theorem shaded_percentage_of_grid :
  percent_shaded 36 16 = 44.44 :=
by 
  sorry

end shaded_percentage_of_grid_l575_57585


namespace faster_train_speed_l575_57565

theorem faster_train_speed (length_train : ℝ) (time_cross : ℝ) (speed_ratio : ℝ) (total_distance : ℝ) (relative_speed : ℝ) :
  length_train = 100 → 
  time_cross = 8 → 
  speed_ratio = 2 → 
  total_distance = 2 * length_train → 
  relative_speed = (1 + speed_ratio) * (total_distance / time_cross) → 
  (1 + speed_ratio) * (total_distance / time_cross) / 3 * 2 = 8.33 := 
by
  intros
  sorry

end faster_train_speed_l575_57565


namespace pond_width_l575_57556

theorem pond_width
  (L : ℝ) (D : ℝ) (V : ℝ) (W : ℝ)
  (hL : L = 20)
  (hD : D = 5)
  (hV : V = 1000)
  (hVolume : V = L * W * D) :
  W = 10 :=
by {
  sorry
}

end pond_width_l575_57556


namespace equilateral_triangle_perimeter_l575_57597

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_l575_57597


namespace lucy_money_left_l575_57573

theorem lucy_money_left : 
  ∀ (initial_money : ℕ) 
    (one_third_loss : ℕ → ℕ) 
    (one_fourth_spend : ℕ → ℕ), 
    initial_money = 30 → 
    one_third_loss initial_money = initial_money / 3 → 
    one_fourth_spend (initial_money - one_third_loss initial_money) = (initial_money - one_third_loss initial_money) / 4 → 
  initial_money - one_third_loss initial_money - one_fourth_spend (initial_money - one_third_loss initial_money) = 15 :=
by
  intros initial_money one_third_loss one_fourth_spend
  intro h_initial_money
  intro h_one_third_loss
  intro h_one_fourth_spend
  sorry

end lucy_money_left_l575_57573


namespace find_missing_number_l575_57551

theorem find_missing_number (x : ℚ) (h : 11 * x + 4 = 7) : x = 9 / 11 :=
sorry

end find_missing_number_l575_57551


namespace present_age_of_B_l575_57575

-- Definitions
variables (a b : ℕ)

-- Conditions
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 7

-- Theorem to prove
theorem present_age_of_B (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 37 := by
  sorry

end present_age_of_B_l575_57575


namespace tina_spent_on_books_l575_57531

theorem tina_spent_on_books : 
  ∀ (saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left : ℤ),
  saved_in_june = 27 →
  saved_in_july = 14 →
  saved_in_august = 21 →
  spend_on_shoes = 17 →
  money_left = 40 →
  (saved_in_june + saved_in_july + saved_in_august) - spend_on_books - spend_on_shoes = money_left →
  spend_on_books = 5 :=
by
  intros saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left
  intros h_june h_july h_august h_shoes h_money_left h_eq
  sorry

end tina_spent_on_books_l575_57531


namespace not_possible_to_cover_l575_57558

namespace CubeCovering

-- Defining the cube and its properties
def cube_side_length : ℕ := 4
def face_area := cube_side_length * cube_side_length
def total_faces : ℕ := 6
def faces_to_cover : ℕ := 3

-- Defining the paper strips and their properties
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def strip_area := strip_length * strip_width
def num_strips : ℕ := 16

-- Calculate the total area to cover
def total_area_to_cover := faces_to_cover * face_area
def total_area_strips := num_strips * strip_area

-- Statement: Prove that it is not possible to cover the three faces
theorem not_possible_to_cover : total_area_to_cover = 48 → total_area_strips = 48 → false := by
  intro h1 h2
  sorry

end CubeCovering

end not_possible_to_cover_l575_57558


namespace problem_statement_l575_57599

variables {α β : Plane} {m : Line}

def parallel (a b : Plane) : Prop := sorry
def perpendicular (m : Line) (π : Plane) : Prop := sorry

axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_trans {m : Line} {a b : Plane} : perpendicular m a → parallel a b → perpendicular m b

theorem problem_statement (h1 : parallel α β) (h2 : perpendicular m α) : perpendicular m β :=
  perpendicular_trans h2 (parallel_symm h1)

end problem_statement_l575_57599


namespace manager_salary_is_3600_l575_57593

-- Definitions based on the conditions
def average_salary_20_employees := 1500
def number_of_employees := 20
def new_average_salary := 1600
def number_of_people_incl_manager := number_of_employees + 1

-- Calculate necessary total salaries and manager's salary
def total_salary_of_20_employees := number_of_employees * average_salary_20_employees
def new_total_salary_with_manager := number_of_people_incl_manager * new_average_salary
def manager_monthly_salary := new_total_salary_with_manager - total_salary_of_20_employees

-- The statement to be proved
theorem manager_salary_is_3600 : manager_monthly_salary = 3600 :=
by
  sorry

end manager_salary_is_3600_l575_57593


namespace inequality_proof_equality_condition_l575_57579

variables {a b c x y z : ℕ}

theorem inequality_proof (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 ≤ (c + z) ^ 2 :=
sorry

theorem equality_condition (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 = (c + z) ^ 2 ↔ a * z = c * x ∧ a * y = b * x :=
sorry

end inequality_proof_equality_condition_l575_57579


namespace simplify_and_evaluate_l575_57523

def expr (a b : ℤ) := -a^2 * b + (3 * a * b^2 - a^2 * b) - 2 * (2 * a * b^2 - a^2 * b)

theorem simplify_and_evaluate : expr (-1) (-2) = -4 := by
  sorry

end simplify_and_evaluate_l575_57523


namespace cloth_length_l575_57555

theorem cloth_length (L : ℕ) (x : ℕ) :
  32 + x = L ∧ 20 + 3 * x = L → L = 38 :=
by
  sorry

end cloth_length_l575_57555


namespace profit_percentage_with_discount_is_26_l575_57571

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_without_discount : ℝ := 31.25
noncomputable def discount_percentage : ℝ := 4

noncomputable def selling_price_without_discount : ℝ :=
  cost_price * (1 + profit_percentage_without_discount / 100)

noncomputable def discount : ℝ := 
  discount_percentage / 100 * selling_price_without_discount

noncomputable def selling_price_with_discount : ℝ :=
  selling_price_without_discount - discount

noncomputable def profit_with_discount : ℝ := 
  selling_price_with_discount - cost_price

noncomputable def profit_percentage_with_discount : ℝ := 
  (profit_with_discount / cost_price) * 100

theorem profit_percentage_with_discount_is_26 :
  profit_percentage_with_discount = 26 := by 
  sorry

end profit_percentage_with_discount_is_26_l575_57571


namespace probability_blue_or_purple_is_4_over_11_l575_57508

def total_jelly_beans : ℕ := 10 + 12 + 13 + 15 + 5
def blue_or_purple_jelly_beans : ℕ := 15 + 5
def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_4_over_11 :
  probability_blue_or_purple = 4 / 11 :=
sorry

end probability_blue_or_purple_is_4_over_11_l575_57508


namespace factorial_div_sub_factorial_equality_l575_57592

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end factorial_div_sub_factorial_equality_l575_57592


namespace unbroken_seashells_l575_57576

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) (h1 : total_seashells = 23) (h2 : broken_seashells = 11) : total_seashells - broken_seashells = 12 := by
  sorry

end unbroken_seashells_l575_57576


namespace total_amount_is_correct_l575_57537

-- Given conditions
def original_price : ℝ := 200
def discount_rate: ℝ := 0.25
def coupon_value: ℝ := 10
def tax_rate: ℝ := 0.05

-- Define the price calculations
def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
def price_after_coupon (p : ℝ) (c : ℝ) : ℝ := p - c
def final_price_with_tax (p : ℝ) (t : ℝ) : ℝ := p * (1 + t)

-- Goal: Prove the final amount the customer pays
theorem total_amount_is_correct : final_price_with_tax (price_after_coupon (discounted_price original_price discount_rate) coupon_value) tax_rate = 147 := by
  sorry

end total_amount_is_correct_l575_57537


namespace pants_cost_l575_57546

theorem pants_cost (starting_amount shirts_cost shirts_count amount_left money_after_shirts pants_cost : ℕ) 
    (h1 : starting_amount = 109)
    (h2 : shirts_cost = 11)
    (h3 : shirts_count = 2)
    (h4 : amount_left = 74)
    (h5 : money_after_shirts = starting_amount - shirts_cost * shirts_count)
    (h6 : pants_cost = money_after_shirts - amount_left) :
  pants_cost = 13 :=
by
  sorry

end pants_cost_l575_57546


namespace correct_statements_l575_57567

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l575_57567


namespace rational_power_sum_l575_57587

theorem rational_power_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = - b) : a ^ 2007 + b ^ 2007 = 1 ∨ a ^ 2007 + b ^ 2007 = -1 := by
  sorry

end rational_power_sum_l575_57587


namespace point_D_coordinates_l575_57506

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (-1, 5)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def scalar_mul (k : ℝ) (v : point) : point := (k * v.1, k * v.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)

def D : point := vector_add A (scalar_mul 3 (vector_sub B A))

theorem point_D_coordinates : D = (-7, 9) :=
by
  -- Proof goes here
  sorry

end point_D_coordinates_l575_57506


namespace minimum_xy_minimum_x_plus_y_l575_57520

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
sorry

theorem minimum_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end minimum_xy_minimum_x_plus_y_l575_57520


namespace puzzles_pieces_count_l575_57503

theorem puzzles_pieces_count :
  let pieces_per_hour := 100
  let hours_per_day := 7
  let days := 7
  let total_pieces_can_put_together := pieces_per_hour * hours_per_day * days
  let pieces_per_puzzle1 := 300
  let number_of_puzzles1 := 8
  let total_pieces_puzzles1 := pieces_per_puzzle1 * number_of_puzzles1
  let remaining_pieces := total_pieces_can_put_together - total_pieces_puzzles1
  let number_of_puzzles2 := 5
  remaining_pieces / number_of_puzzles2 = 500
:= by
  sorry

end puzzles_pieces_count_l575_57503


namespace swimmer_speed_proof_l575_57568

-- Definition of the conditions
def current_speed : ℝ := 2
def swimming_time : ℝ := 1.5
def swimming_distance : ℝ := 3

-- Prove: Swimmer's speed in still water
def swimmer_speed_in_still_water : ℝ := 4

-- Statement: Given the conditions, the swimmer's speed in still water equals 4 km/h
theorem swimmer_speed_proof :
  (swimming_distance = (swimmer_speed_in_still_water - current_speed) * swimming_time) →
  swimmer_speed_in_still_water = 4 :=
by
  intro h
  sorry

end swimmer_speed_proof_l575_57568


namespace arithmetic_seq_a6_l575_57566

variable (a : ℕ → ℝ)

-- Conditions
axiom a3 : a 3 = 16
axiom a9 : a 9 = 80

-- Theorem to prove
theorem arithmetic_seq_a6 : a 6 = 48 :=
by
  sorry

end arithmetic_seq_a6_l575_57566


namespace negation_proposition_l575_57542

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :
  ¬(∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) = ∃ x : ℝ, x^2 - 2*x + 4 > 0 :=
by
  sorry

end negation_proposition_l575_57542


namespace hours_of_use_per_charge_l575_57532

theorem hours_of_use_per_charge
  (c h u : ℕ)
  (h_c : c = 10)
  (h_fraction : h = 6)
  (h_use : 6 * u = 12) :
  u = 2 :=
sorry

end hours_of_use_per_charge_l575_57532


namespace frac_pow_eq_l575_57594

theorem frac_pow_eq : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by 
  sorry

end frac_pow_eq_l575_57594


namespace factor_difference_of_squares_l575_57540

theorem factor_difference_of_squares (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := 
sorry

end factor_difference_of_squares_l575_57540


namespace xiaoming_grandfather_age_l575_57539

-- Define the conditions
def age_cond (x : ℕ) : Prop :=
  ((x - 15) / 4 - 6) * 10 = 100

-- State the problem
theorem xiaoming_grandfather_age (x : ℕ) (h : age_cond x) : x = 79 := 
sorry

end xiaoming_grandfather_age_l575_57539


namespace walking_rate_ratio_l575_57526

theorem walking_rate_ratio (R R' : ℝ)
  (h : R * 36 = R' * 32) : R' / R = 9 / 8 :=
sorry

end walking_rate_ratio_l575_57526


namespace part1_part2_l575_57563

theorem part1 (x : ℝ) : |x + 3| - 2 * x - 1 < 0 → 2 < x :=
by sorry

theorem part2 (m : ℝ) : (m > 0) →
  (∃ x : ℝ, |x - m| + |x + 1/m| = 2) → m = 1 :=
by sorry

end part1_part2_l575_57563


namespace new_ratio_milk_water_after_adding_milk_l575_57502

variable (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ)
variable (added_milk_volume : ℕ)

def ratio_of_mix_after_addition (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) 
  (added_milk_volume : ℕ) : ℕ × ℕ :=
  let total_parts := initial_milk_ratio + initial_water_ratio
  let part_volume := initial_volume / total_parts
  let initial_milk_volume := initial_milk_ratio * part_volume
  let initial_water_volume := initial_water_ratio * part_volume
  let new_milk_volume := initial_milk_volume + added_milk_volume
  (new_milk_volume / initial_water_volume, 1)

theorem new_ratio_milk_water_after_adding_milk 
  (h_initial_volume : initial_volume = 20)
  (h_initial_milk_ratio : initial_milk_ratio = 3)
  (h_initial_water_ratio : initial_water_ratio = 1)
  (h_added_milk_volume : added_milk_volume = 5) : 
  ratio_of_mix_after_addition initial_volume initial_milk_ratio initial_water_ratio added_milk_volume = (4, 1) :=
  by
    sorry

end new_ratio_milk_water_after_adding_milk_l575_57502


namespace maximum_real_roots_maximum_total_real_roots_l575_57515

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

def quadratic_discriminant (p q r : ℝ) : ℝ := q^2 - 4 * p * r

theorem maximum_real_roots (h1 : quadratic_discriminant a b c < 0)
  (h2 : quadratic_discriminant b c a < 0)
  (h3 : quadratic_discriminant c a b < 0) :
  ∀ (x : ℝ), (a * x^2 + b * x + c ≠ 0) ∧ 
             (b * x^2 + c * x + a ≠ 0) ∧ 
             (c * x^2 + a * x + b ≠ 0) :=
sorry

theorem maximum_total_real_roots :
    ∃ x : ℝ, ∃ y : ℝ, ∃ z : ℝ,
    (a * x^2 + b * x + c = 0) ∧
    (b * y^2 + c * y + a = 0) ∧
    (a * y ≠ x) ∧
    (c * z^2 + a * z + b = 0) ∧
    (b * z ≠ x) ∧
    (c * z ≠ y) :=
sorry

end maximum_real_roots_maximum_total_real_roots_l575_57515


namespace conditional_probability_of_wind_given_rain_l575_57533

theorem conditional_probability_of_wind_given_rain (P_A P_B P_A_and_B : ℚ)
  (h1: P_A = 4/15) (h2: P_B = 2/15) (h3: P_A_and_B = 1/10) :
  P_A_and_B / P_A = 3/8 :=
by
  sorry

end conditional_probability_of_wind_given_rain_l575_57533


namespace commercial_break_duration_l575_57527

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l575_57527


namespace person_B_D_coins_l575_57512

theorem person_B_D_coins
  (a d : ℤ)
  (h1 : a - 3 * d = 58)
  (h2 : a - 2 * d = 58)
  (h3 : a + d = 60)
  (h4 : a + 2 * d = 60)
  (h5 : a + 3 * d = 60) :
  (a - 2 * d = 28) ∧ (a = 24) :=
by
  sorry

end person_B_D_coins_l575_57512


namespace singing_only_pupils_l575_57586

theorem singing_only_pupils (total_pupils debate_only both : ℕ) (h1 : total_pupils = 55) (h2 : debate_only = 10) (h3 : both = 17) :
  total_pupils - debate_only = 45 :=
by
  -- skipping proof
  sorry

end singing_only_pupils_l575_57586


namespace interval_for_f_l575_57580

noncomputable def f (x : ℝ) : ℝ :=
-0.5 * x ^ 2 + 13 / 2

theorem interval_for_f (a b : ℝ) :
  f a = 2 * b ∧ f b = 2 * a ∧ (a ≤ 0 ∨ 0 ≤ b) → 
  ([a, b] = [1, 3] ∨ [a, b] = [-2 - Real.sqrt 17, 13 / 4]) :=
by sorry

end interval_for_f_l575_57580


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l575_57559

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l575_57559


namespace value_of_f_m_plus_one_is_negative_l575_57554

-- Definitions for function and condition
def f (x a : ℝ) := x^2 - x + a 

-- Problem statement: Given that 'f(-m) < 0', prove 'f(m+1) < 0'
theorem value_of_f_m_plus_one_is_negative (a m : ℝ) (h : f (-m) a < 0) : f (m + 1) a < 0 :=
by 
  sorry

end value_of_f_m_plus_one_is_negative_l575_57554


namespace avg_visitors_proof_l575_57589

-- Define the constants and conditions
def Sundays_visitors : ℕ := 500
def total_days : ℕ := 30
def avg_visitors_per_day : ℕ := 200

-- Total visits on Sundays within the month
def visits_on_Sundays := 5 * Sundays_visitors

-- Total visitors for the month
def total_visitors := total_days * avg_visitors_per_day

-- Average visitors on other days (Monday to Saturday)
def avg_visitors_other_days : ℕ :=
  (total_visitors - visits_on_Sundays) / (total_days - 5)

-- The theorem stating the problem and corresponding answer
theorem avg_visitors_proof (V : ℕ) 
  (h1 : Sundays_visitors = 500)
  (h2 : total_days = 30)
  (h3 : avg_visitors_per_day = 200)
  (h4 : visits_on_Sundays = 5 * Sundays_visitors)
  (h5 : total_visitors = total_days * avg_visitors_per_day)
  (h6 : avg_visitors_other_days = (total_visitors - visits_on_Sundays) / (total_days - 5))
  : V = 140 :=
by
  -- Proof is not required, just state the theorem
  sorry

end avg_visitors_proof_l575_57589


namespace value_of_c_l575_57557

theorem value_of_c (a b c : ℚ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 7) (h3 : a - b + 3 = c - 2 * b) : c = 21 / 2 :=
sorry

end value_of_c_l575_57557


namespace simplify_polynomial_l575_57545

def P (x : ℝ) : ℝ := 3*x^3 + 4*x^2 - 5*x + 8
def Q (x : ℝ) : ℝ := 2*x^3 + x^2 + 3*x - 15

theorem simplify_polynomial (x : ℝ) : P x - Q x = x^3 + 3*x^2 - 8*x + 23 := 
by 
  -- proof goes here
  sorry

end simplify_polynomial_l575_57545
