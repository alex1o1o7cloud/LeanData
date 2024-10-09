import Mathlib

namespace move_symmetric_point_left_l1437_143770

-- Define the original point and the operations
def original_point : ℝ × ℝ := (-2, 3)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Prove the resulting point after the operations
theorem move_symmetric_point_left : move_left (symmetric_point original_point) 2 = (0, -3) :=
by
  sorry

end move_symmetric_point_left_l1437_143770


namespace tan_increasing_interval_l1437_143745

noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  {x | (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12)}

theorem tan_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12) ↔ 
    (∃ y, y = (2 * x + Real.pi / 3) ∧ Real.tan y > Real.tan (2 * x + Real.pi / 3 - 1e-6)) :=
sorry

end tan_increasing_interval_l1437_143745


namespace evaluate_ninth_roots_of_unity_product_l1437_143765

theorem evaluate_ninth_roots_of_unity_product : 
  (3 - Complex.exp (2 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (4 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (6 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (8 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (10 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (12 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (14 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (16 * Real.pi * Complex.I / 9)) 
  = 9841 := 
by 
  sorry

end evaluate_ninth_roots_of_unity_product_l1437_143765


namespace length_width_difference_l1437_143780

noncomputable def width : ℝ := Real.sqrt (588 / 8)
noncomputable def length : ℝ := 4 * width
noncomputable def difference : ℝ := length - width

theorem length_width_difference : difference = 25.722 := by
  sorry

end length_width_difference_l1437_143780


namespace cards_choice_ways_l1437_143776

theorem cards_choice_ways (S : List Char) (cards : Finset (Char × ℕ)) :
  (∀ c ∈ cards, c.1 ∈ S) ∧
  (∀ (c1 c2 : Char × ℕ), c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.1 ≠ c2.1) ∧
  (∃ c ∈ cards, c.2 = 1 ∧ c.1 = 'H') →
  (∃ c ∈ cards, c.2 = 1) →
  ∃ (ways : ℕ), ways = 1014 := 
sorry

end cards_choice_ways_l1437_143776


namespace compound_interest_time_period_l1437_143733

theorem compound_interest_time_period (P r I : ℝ) (n A t : ℝ) 
(hP : P = 6000) 
(hr : r = 0.10) 
(hI : I = 1260.000000000001) 
(hn : n = 1)
(hA : A = P + I)
(ht_eqn: (A / P) = (1 + r / n) ^ t) :
t = 2 := 
by sorry

end compound_interest_time_period_l1437_143733


namespace morning_rowers_count_l1437_143797

def number_afternoon_rowers : ℕ := 7
def total_rowers : ℕ := 60

def number_morning_rowers : ℕ :=
  total_rowers - number_afternoon_rowers

theorem morning_rowers_count :
  number_morning_rowers = 53 := by
  sorry

end morning_rowers_count_l1437_143797


namespace exists_subset_sum_2n_l1437_143701

theorem exists_subset_sum_2n (n : ℕ) (h : n > 3) (s : Finset ℕ)
  (hs : ∀ x ∈ s, x < 2 * n) (hs_card : s.card = 2 * n)
  (hs_sum : s.sum id = 4 * n) :
  ∃ t ⊆ s, t.sum id = 2 * n :=
by sorry

end exists_subset_sum_2n_l1437_143701


namespace maxwell_walking_speed_l1437_143747

open Real

theorem maxwell_walking_speed (v : ℝ) : 
  (∀ (v : ℝ), (4 * v + 6 * 3 = 34)) → v = 4 :=
by
  intros
  have h1 : 4 * v + 18 = 34 := by sorry
  have h2 : 4 * v = 16 := by sorry
  have h3 : v = 4 := by sorry
  exact h3

end maxwell_walking_speed_l1437_143747


namespace cyclic_sum_inequality_l1437_143751

theorem cyclic_sum_inequality
  (a b c d e : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end cyclic_sum_inequality_l1437_143751


namespace xy_sum_is_2_l1437_143792

theorem xy_sum_is_2 (x y : ℝ) (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := 
by
  sorry

end xy_sum_is_2_l1437_143792


namespace xiaodong_sister_age_correct_l1437_143775

/-- Let's define the conditions as Lean definitions -/
def sister_age := 13
def xiaodong_age := sister_age - 8
def sister_age_in_3_years := sister_age + 3
def xiaodong_age_in_3_years := xiaodong_age + 3

/-- We need to prove that in 3 years, the sister's age will be twice Xiaodong's age -/
theorem xiaodong_sister_age_correct :
  (sister_age_in_3_years = 2 * xiaodong_age_in_3_years) → sister_age = 13 :=
by
  sorry

end xiaodong_sister_age_correct_l1437_143775


namespace water_needed_quarts_l1437_143727

-- Definitions from conditions
def ratio_water : ℕ := 8
def ratio_lemon : ℕ := 1
def total_gallons : ℚ := 1.5
def gallons_to_quarts : ℚ := 4

-- State what needs to be proven
theorem water_needed_quarts : 
  (total_gallons * gallons_to_quarts * (ratio_water / (ratio_water + ratio_lemon))) = 16 / 3 :=
by
  sorry

end water_needed_quarts_l1437_143727


namespace sin_A_plus_B_lt_sin_A_add_sin_B_l1437_143748

variable {A B : ℝ}
variable (A_pos : 0 < A)
variable (B_pos : 0 < B)
variable (AB_sum_pi : A + B < π)

theorem sin_A_plus_B_lt_sin_A_add_sin_B (a b : ℝ) (h1 : a = Real.sin (A + B)) (h2 : b = Real.sin A + Real.sin B) : 
  a < b := by
  sorry

end sin_A_plus_B_lt_sin_A_add_sin_B_l1437_143748


namespace sampling_probabilities_equal_l1437_143730

-- Definitions according to the problem conditions
def population_size := ℕ
def sample_size := ℕ
def simple_random_sampling (N n : ℕ) : Prop := sorry
def systematic_sampling (N n : ℕ) : Prop := sorry
def stratified_sampling (N n : ℕ) : Prop := sorry

-- Probabilities
def P1 : ℝ := sorry -- Probability for simple random sampling
def P2 : ℝ := sorry -- Probability for systematic sampling
def P3 : ℝ := sorry -- Probability for stratified sampling

-- Each definition directly corresponds to a condition in the problem statement.
-- Now, we summarize the equivalent proof problem in Lean.

theorem sampling_probabilities_equal (N n : ℕ) (h1 : simple_random_sampling N n) (h2 : systematic_sampling N n) (h3 : stratified_sampling N n) :
  P1 = P2 ∧ P2 = P3 :=
by sorry

end sampling_probabilities_equal_l1437_143730


namespace functional_equation_g_l1437_143746

variable (g : ℝ → ℝ)
variable (f : ℝ)
variable (h : ℝ)

theorem functional_equation_g (H1 : ∀ x y : ℝ, g (x + y) = g x * g y)
                            (H2 : g 3 = 4) :
                            g 6 = 16 := 
by
  sorry

end functional_equation_g_l1437_143746


namespace democrats_ratio_l1437_143714

variable (F M D_F D_M TotalParticipants : ℕ)

-- Assume the following conditions
variables (H1 : F + M = 660)
variables (H2 : D_F = 1 / 2 * F)
variables (H3 : D_F = 110)
variables (H4 : D_M = 1 / 4 * M)
variables (H5 : TotalParticipants = 660)

theorem democrats_ratio 
  (H1 : F + M = 660)
  (H2 : D_F = 1 / 2 * F)
  (H3 : D_F = 110)
  (H4 : D_M = 1 / 4 * M)
  (H5 : TotalParticipants = 660) :
  (D_F + D_M) / TotalParticipants = 1 / 3
:= 
  sorry

end democrats_ratio_l1437_143714


namespace empty_solution_set_implies_a_range_l1437_143787

def f (a x: ℝ) := x^2 + (1 - a) * x - a

theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬ (f a (f a x) < 0)) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by
  sorry

end empty_solution_set_implies_a_range_l1437_143787


namespace correct_equation_l1437_143709

theorem correct_equation (x : ℝ) : (-x^2)^2 = x^4 := by sorry

end correct_equation_l1437_143709


namespace amelia_drove_tuesday_l1437_143707

-- Define the known quantities
def total_distance : ℕ := 8205
def distance_monday : ℕ := 907
def remaining_distance : ℕ := 6716

-- Define the distance driven on Tuesday and state the theorem
def distance_tuesday : ℕ := total_distance - (distance_monday + remaining_distance)

-- Theorem stating the distance driven on Tuesday is 582 kilometers
theorem amelia_drove_tuesday : distance_tuesday = 582 := 
by
  -- We skip the proof for now
  sorry

end amelia_drove_tuesday_l1437_143707


namespace sufficient_but_not_necessary_condition_l1437_143789

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ (|x| > 1 → (x > 1 ∨ x < -1)) ∧ ¬(|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1437_143789


namespace bird_needs_more_twigs_l1437_143785

variable (base_twigs : ℕ := 12)
variable (additional_twigs_per_base : ℕ := 6)
variable (fraction_dropped : ℚ := 1/3)

theorem bird_needs_more_twigs (tree_dropped : ℕ) : 
  tree_dropped = (additional_twigs_per_base * base_twigs) * 1/3 →
  (base_twigs * additional_twigs_per_base - tree_dropped) = 48 :=
by
  sorry

end bird_needs_more_twigs_l1437_143785


namespace ratio_hexagon_octagon_l1437_143753

noncomputable def ratio_of_areas (s : ℝ) :=
  let A1 := s / (2 * Real.tan (Real.pi / 6))
  let H1 := s / (2 * Real.sin (Real.pi / 6))
  let area1 := Real.pi * (H1^2 - A1^2)
  let A2 := s / (2 * Real.tan (Real.pi / 8))
  let H2 := s / (2 * Real.sin (Real.pi / 8))
  let area2 := Real.pi * (H2^2 - A2^2)
  area1 / area2

theorem ratio_hexagon_octagon (s : ℝ) (h : s = 3) : ratio_of_areas s = 49 / 25 :=
  sorry

end ratio_hexagon_octagon_l1437_143753


namespace series_remainder_is_zero_l1437_143718

theorem series_remainder_is_zero :
  let a : ℕ := 4
  let d : ℕ := 6
  let n : ℕ := 17
  let l : ℕ := a + d * (n - 1) -- last term
  let S : ℕ := n * (a + l) / 2 -- sum of the series
  S % 17 = 0 := by
  sorry

end series_remainder_is_zero_l1437_143718


namespace maya_total_pages_read_l1437_143722

def last_week_books : ℕ := 5
def pages_per_book : ℕ := 300
def this_week_multiplier : ℕ := 2

theorem maya_total_pages_read : 
  (last_week_books * pages_per_book * (1 + this_week_multiplier)) = 4500 :=
by
  sorry

end maya_total_pages_read_l1437_143722


namespace arithmetic_seq_sum_l1437_143724

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 5 + a 6 + a 7 = 1) : a 3 + a 9 = 2 / 3 :=
sorry

end arithmetic_seq_sum_l1437_143724


namespace train_speed_proof_l1437_143721

def identical_trains_speed : Real :=
  11.11

theorem train_speed_proof :
  ∀ (v : ℝ),
  (∀ (t t' : ℝ), 
  (t = 150 / v) ∧ 
  (t' = 300 / v) ∧ 
  ((t' + 100 / v) = 36)) → v = identical_trains_speed :=
by
  sorry

end train_speed_proof_l1437_143721


namespace rectangle_area_l1437_143793

/-- A figure is formed by a triangle and a rectangle, using 60 equal sticks.
Each side of the triangle uses 6 sticks, and each stick measures 5 cm in length.
Prove that the area of the rectangle is 2250 cm². -/
theorem rectangle_area (sticks_total : ℕ) (sticks_per_side_triangle : ℕ) (stick_length_cm : ℕ)
    (sticks_used_triangle : ℕ) (sticks_left_rectangle : ℕ) (sticks_per_width_rectangle : ℕ)
    (width_sticks_rectangle : ℕ) (length_sticks_rectangle : ℕ) (width_cm : ℕ) (length_cm : ℕ)
    (area_rectangle : ℕ) 
    (h_sticks_total : sticks_total = 60)
    (h_sticks_per_side_triangle : sticks_per_side_triangle = 6)
    (h_stick_length_cm : stick_length_cm = 5)
    (h_sticks_used_triangle  : sticks_used_triangle = sticks_per_side_triangle * 3)
    (h_sticks_left_rectangle : sticks_left_rectangle = sticks_total - sticks_used_triangle)
    (h_sticks_per_width_rectangle : sticks_per_width_rectangle = 6 * 2) 
    (h_width_sticks_rectangle : width_sticks_rectangle = 6)
    (h_length_sticks_rectangle : length_sticks_rectangle = (sticks_left_rectangle - sticks_per_width_rectangle) / 2)
    (h_width_cm : width_cm = width_sticks_rectangle * stick_length_cm)
    (h_length_cm : length_cm = length_sticks_rectangle * stick_length_cm)
    (h_area_rectangle : area_rectangle = width_cm * length_cm) :
    area_rectangle = 2250 := 
by sorry

end rectangle_area_l1437_143793


namespace renata_donation_l1437_143781

variable (D L : ℝ)

theorem renata_donation : ∃ D : ℝ, 
  (10 - D + 90 - L - 2 + 65 = 94) ↔ D = 4 :=
by
  sorry

end renata_donation_l1437_143781


namespace largest_n_for_factored_quad_l1437_143710

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l1437_143710


namespace weight_of_each_bag_of_flour_l1437_143741

-- Definitions based on the given conditions
def cookies_eaten_by_Jim : ℕ := 15
def cookies_left : ℕ := 105
def total_cookies : ℕ := cookies_eaten_by_Jim + cookies_left

def cookies_per_dozen : ℕ := 12
def pounds_per_dozen : ℕ := 2

def dozens_of_cookies := total_cookies / cookies_per_dozen
def total_pounds_of_flour := dozens_of_cookies * pounds_per_dozen

def bags_of_flour : ℕ := 4

-- Question to be proved
theorem weight_of_each_bag_of_flour : total_pounds_of_flour / bags_of_flour = 5 := by
  sorry

end weight_of_each_bag_of_flour_l1437_143741


namespace toy_cost_l1437_143778

-- Definitions based on the conditions in part a)
def initial_amount : ℕ := 57
def spent_amount : ℕ := 49
def remaining_amount : ℕ := initial_amount - spent_amount
def number_of_toys : ℕ := 2

-- Statement to prove that each toy costs 4 dollars
theorem toy_cost :
  (remaining_amount / number_of_toys) = 4 :=
by
  sorry

end toy_cost_l1437_143778


namespace min_time_one_ball_l1437_143723

noncomputable def children_circle_min_time (n : ℕ) := 98

theorem min_time_one_ball (n : ℕ) (h1 : n = 99) : 
  children_circle_min_time n = 98 := 
by 
  sorry

end min_time_one_ball_l1437_143723


namespace probability_correct_guesses_l1437_143704

theorem probability_correct_guesses:
  let p_wrong := (5/6 : ℚ)
  let p_miss_all := p_wrong ^ 5
  let p_at_least_one_correct := 1 - p_miss_all
  p_at_least_one_correct = 4651/7776 := by
  sorry

end probability_correct_guesses_l1437_143704


namespace initial_number_of_boarders_l1437_143703

theorem initial_number_of_boarders (B D : ℕ) (h1 : B / D = 2 / 5) (h2 : (B + 15) / D = 1 / 2) : B = 60 :=
by
  -- Proof needs to be provided here
  sorry

end initial_number_of_boarders_l1437_143703


namespace smallest_possible_AAB_l1437_143716

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end smallest_possible_AAB_l1437_143716


namespace base_of_second_term_l1437_143726

theorem base_of_second_term (e : ℕ) (base : ℝ) 
  (h1 : e = 35) 
  (h2 : (1/5)^e * base^18 = 1 / (2 * (10)^35)) : 
  base = 1/4 :=
by
  sorry

end base_of_second_term_l1437_143726


namespace second_person_days_l1437_143743

theorem second_person_days (h1 : 2 * (1 : ℝ) / 8 = 1) 
                           (h2 : 1 / 24 + x / 24 = 1 / 8) : x = 1 / 12 :=
sorry

end second_person_days_l1437_143743


namespace LCM_of_fractions_l1437_143734

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l1437_143734


namespace unique_real_solution_l1437_143798

noncomputable def cubic_eq (b x : ℝ) : ℝ :=
  x^3 - b * x^2 - 3 * b * x + b^2 - 2

theorem unique_real_solution (b : ℝ) :
  (∃! x : ℝ, cubic_eq b x = 0) ↔ b = 7 / 4 :=
by
  sorry

end unique_real_solution_l1437_143798


namespace find_quadruple_l1437_143737

theorem find_quadruple :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a^3 + b^4 + c^5 = d^11 ∧ a * b * c < 10^5 :=
sorry

end find_quadruple_l1437_143737


namespace element_in_set_l1437_143779

theorem element_in_set : 1 ∈ ({0, 1} : Set ℕ) := 
by 
  -- Proof goes here
  sorry

end element_in_set_l1437_143779


namespace min_quadratic_expression_l1437_143767

theorem min_quadratic_expression:
  ∀ x : ℝ, x = 3 → (x^2 - 6 * x + 5 = -4) :=
by
  sorry

end min_quadratic_expression_l1437_143767


namespace simplify_expression_l1437_143729

variable (x : ℝ)

theorem simplify_expression : (5 * x + 2 * (4 + x)) = (7 * x + 8) := 
by
  sorry

end simplify_expression_l1437_143729


namespace range_of_a_l1437_143768

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := {x | abs (x - 2) ≤ a}
def set_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

lemma disjoint_sets (A B : Set ℝ) : A ∩ B = ∅ :=
  sorry

theorem range_of_a (h : set_A a ∩ set_B = ∅) : a < 1 :=
  by
  sorry

end range_of_a_l1437_143768


namespace gasoline_amount_added_l1437_143772

noncomputable def initial_fill (capacity : ℝ) : ℝ := (3 / 4) * capacity
noncomputable def final_fill (capacity : ℝ) : ℝ := (9 / 10) * capacity
noncomputable def gasoline_added (capacity : ℝ) : ℝ := final_fill capacity - initial_fill capacity

theorem gasoline_amount_added :
  ∀ (capacity : ℝ), capacity = 24 → gasoline_added capacity = 3.6 :=
  by
    intros capacity h
    rw [h]
    have initial_fill_24 : initial_fill 24 = 18 := by norm_num [initial_fill]
    have final_fill_24 : final_fill 24 = 21.6 := by norm_num [final_fill]
    have gasoline_added_24 : gasoline_added 24 = 3.6 :=
      by rw [gasoline_added, initial_fill_24, final_fill_24]; norm_num
    exact gasoline_added_24

end gasoline_amount_added_l1437_143772


namespace total_cats_in_meow_and_paw_l1437_143761

-- Define the conditions
def CatsInCatCafeCool : Nat := 5
def CatsInCatCafePaw : Nat := 2 * CatsInCatCafeCool
def CatsInCatCafeMeow : Nat := 3 * CatsInCatCafePaw

-- Define the total number of cats in Cat Cafe Meow and Cat Cafe Paw
def TotalCats : Nat := CatsInCatCafeMeow + CatsInCatCafePaw

-- The theorem stating the problem
theorem total_cats_in_meow_and_paw : TotalCats = 40 :=
by
  sorry

end total_cats_in_meow_and_paw_l1437_143761


namespace podcast_length_l1437_143757

theorem podcast_length (x : ℝ) (hx : x + 2 * x + 1.75 + 1 + 1 = 6) : x = 0.75 :=
by {
  -- We do not need the proof steps here
  sorry
}

end podcast_length_l1437_143757


namespace part1_profit_in_april_part2_price_reduction_l1437_143719

-- Given conditions
def cost_per_bag : ℕ := 16
def original_price_per_bag : ℕ := 30
def reduction_amount : ℕ := 5
def increase_in_sales_rate : ℕ := 20
def original_sales_volume : ℕ := 200
def target_profit : ℕ := 2860

-- Part 1: When the price per bag of noodles is reduced by 5 yuan
def profit_in_april_when_reduced_by_5 (cost_per_bag original_price_per_bag reduction_amount increase_in_sales_rate original_sales_volume : ℕ) : ℕ := 
  let new_price := original_price_per_bag - reduction_amount
  let new_sales_volume := original_sales_volume + (increase_in_sales_rate * reduction_amount)
  let profit_per_bag := new_price - cost_per_bag
  profit_per_bag * new_sales_volume

theorem part1_profit_in_april :
  profit_in_april_when_reduced_by_5 16 30 5 20 200 = 2700 :=
sorry

-- Part 2: Determine the price reduction for a specific target profit
def price_reduction_for_profit (cost_per_bag original_price_per_bag increase_in_sales_rate original_sales_volume target_profit : ℕ) : ℕ :=
  let x := (target_profit - (original_sales_volume * (original_price_per_bag - cost_per_bag))) / (increase_in_sales_rate * (original_price_per_bag - cost_per_bag) - increase_in_sales_rate - original_price_per_bag)
  x

theorem part2_price_reduction :
  price_reduction_for_profit 16 30 20 200 2860 = 3 :=
sorry

end part1_profit_in_april_part2_price_reduction_l1437_143719


namespace solve_equation_l1437_143749

-- Definitions based on the conditions
def equation (a b c d : ℕ) : Prop :=
  2^a * 3^b - 5^c * 7^d = 1

def nonnegative_integers (a b c d : ℕ) : Prop := 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof to show the exact solutions
theorem solve_equation :
  (∃ (a b c d : ℕ), nonnegative_integers a b c d ∧ equation a b c d) ↔ 
  ( (1, 0, 0, 0) = (1, 0, 0, 0) ∨ (3, 0, 0, 1) = (3, 0, 0, 1) ∨ 
    (1, 1, 1, 0) = (1, 1, 1, 0) ∨ (2, 2, 1, 1) = (2, 2, 1, 1) ) := by
  sorry

end solve_equation_l1437_143749


namespace value_of_a_l1437_143744

def star (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem value_of_a (a : ℝ) (h : star a 3 = 15) : a = 11 := 
by
  sorry

end value_of_a_l1437_143744


namespace correct_statements_count_l1437_143795

-- Definitions for each condition
def is_output_correct (stmt : String) : Prop :=
  stmt = "PRINT a, b, c"

def is_input_correct (stmt : String) : Prop :=
  stmt = "INPUT \"x=3\""

def is_assignment_correct_1 (stmt : String) : Prop :=
  stmt = "A=3"

def is_assignment_correct_2 (stmt : String) : Prop :=
  stmt = "A=B ∧ B=C"

-- The main theorem to be proven
theorem correct_statements_count (stmt1 stmt2 stmt3 stmt4 : String) :
  stmt1 = "INPUT a, b, c" → stmt2 = "INPUT x=3" → stmt3 = "3=A" → stmt4 = "A=B=C" →
  (¬ is_output_correct stmt1 ∧ ¬ is_input_correct stmt2 ∧ ¬ is_assignment_correct_1 stmt3 ∧ ¬ is_assignment_correct_2 stmt4) →
  0 = 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_statements_count_l1437_143795


namespace find_k_for_equation_l1437_143705

theorem find_k_for_equation : 
  ∃ k : ℤ, -x^2 - (k + 7) * x - 8 = -(x - 2) * (x - 4) → k = -13 := 
by
  sorry

end find_k_for_equation_l1437_143705


namespace determine_g_l1437_143754

theorem determine_g (t : ℝ) : ∃ (g : ℝ → ℝ), (∀ x y, y = 2 * x - 40 ∧ y = 20 * t - 14 → g t = 10 * t + 13) :=
by
  sorry

end determine_g_l1437_143754


namespace book_sale_revenue_l1437_143706

noncomputable def total_amount_received (price_per_book : ℝ) (B : ℕ) (sold_fraction : ℝ) :=
  sold_fraction * B * price_per_book

theorem book_sale_revenue (B : ℕ) (price_per_book : ℝ) (unsold_books : ℕ) (sold_fraction : ℝ) :
  (1 / 3 : ℝ) * B = unsold_books →
  price_per_book = 3.50 →
  unsold_books = 36 →
  sold_fraction = 2 / 3 →
  total_amount_received price_per_book B sold_fraction = 252 :=
by
  intros h1 h2 h3 h4
  sorry

end book_sale_revenue_l1437_143706


namespace camel_cost_l1437_143784

variables {C H O E G Z : ℕ} 

-- conditions
axiom h1 : 10 * C = 24 * H
axiom h2 : 16 * H = 4 * O
axiom h3 : 6 * O = 4 * E
axiom h4 : 3 * E = 15 * G
axiom h5 : 8 * G = 20 * Z
axiom h6 : 12 * E = 180000

-- goal
theorem camel_cost : C = 6000 :=
by sorry

end camel_cost_l1437_143784


namespace how_many_cubes_needed_l1437_143700

def cube_volume (side_len : ℕ) : ℕ :=
  side_len ^ 3

theorem how_many_cubes_needed (Vsmall Vlarge Vsmall_cube num_small_cubes : ℕ) 
  (h1 : Vsmall = cube_volume 8) 
  (h2 : Vlarge = cube_volume 12) 
  (h3 : Vsmall_cube = cube_volume 2) 
  (h4 : num_small_cubes = (Vlarge - Vsmall) / Vsmall_cube) :
  num_small_cubes = 152 :=
by
  sorry

end how_many_cubes_needed_l1437_143700


namespace integer_satisfies_inequality_l1437_143760

theorem integer_satisfies_inequality (n : ℤ) : 
  (3 : ℚ) / 10 < n / 20 ∧ n / 20 < 2 / 5 → n = 7 :=
sorry

end integer_satisfies_inequality_l1437_143760


namespace evaluate_expression_l1437_143790

theorem evaluate_expression : 
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = (137 / 52) :=
by
  -- We need to evaluate from the innermost part to the outermost,
  -- as noted in the problem statement and solution steps.
  sorry

end evaluate_expression_l1437_143790


namespace correct_quotient_is_48_l1437_143736

theorem correct_quotient_is_48 (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 72 → 
  incorrect_quotient = 24 → 
  correct_divisor = 36 →
  dividend = incorrect_divisor * incorrect_quotient →
  correct_quotient = dividend / correct_divisor →
  correct_quotient = 48 :=
by
  sorry

end correct_quotient_is_48_l1437_143736


namespace rice_and_wheat_grains_division_l1437_143708

-- Definitions for the conditions in the problem
def total_grains : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Proving the approximate amount of wheat grains in the batch  
theorem rice_and_wheat_grains_division : total_grains * (wheat_in_sample / sample_size) = 169 := by 
  sorry

end rice_and_wheat_grains_division_l1437_143708


namespace total_trip_length_l1437_143725

theorem total_trip_length :
  ∀ (d : ℝ), 
    (∀ fuel_per_mile : ℝ, fuel_per_mile = 0.03 →
      ∀ battery_miles : ℝ, battery_miles = 50 →
      ∀ avg_miles_per_gallon : ℝ, avg_miles_per_gallon = 50 →
      (d / (fuel_per_mile * (d - battery_miles))) = avg_miles_per_gallon →
      d = 150) := 
by
  intros d fuel_per_mile fuel_per_mile_eq battery_miles battery_miles_eq avg_miles_per_gallon avg_miles_per_gallon_eq trip_condition
  sorry

end total_trip_length_l1437_143725


namespace employee_B_payment_l1437_143735

theorem employee_B_payment (x : ℝ) (h1 : ∀ A B : ℝ, A + B = 580) (h2 : A = 1.5 * B) : B = 232 :=
by
  sorry

end employee_B_payment_l1437_143735


namespace sequence_sum_l1437_143752

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 1

-- Define the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- State the theorem
theorem sequence_sum : (b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) + b_n (a_n 5) + b_n (a_n 6)) = 126 := by
  sorry

end sequence_sum_l1437_143752


namespace ramesh_paid_price_l1437_143782

theorem ramesh_paid_price {P : ℝ} (h1 : P = 18880 / 1.18) : 
  (0.80 * P + 125 + 250) = 13175 :=
by sorry

end ramesh_paid_price_l1437_143782


namespace pebble_sequence_10_l1437_143712

-- A definition for the sequence based on the given conditions and pattern.
def pebble_sequence : ℕ → ℕ
| 0 => 1
| 1 => 5
| 2 => 12
| 3 => 22
| (n + 4) => pebble_sequence (n + 3) + (3 * (n + 1) + 1)

-- Theorem that states the value at the 10th position in the sequence.
theorem pebble_sequence_10 : pebble_sequence 9 = 145 :=
sorry

end pebble_sequence_10_l1437_143712


namespace probability_X_eq_4_l1437_143771

-- Define the number of students and boys
def total_students := 15
def total_boys := 7
def selected_students := 10

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := n.choose k

-- Calculate the probability
def P_X_eq_4 := (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students

-- The statement to be proven
theorem probability_X_eq_4 :
  P_X_eq_4 = (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students := by
  sorry

end probability_X_eq_4_l1437_143771


namespace min_value_symmetry_l1437_143791

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_symmetry (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic a b c (2 + x) = quadratic a b c (2 - x)) : 
  quadratic a b c 2 < quadratic a b c 1 ∧ quadratic a b c 1 < quadratic a b c 4 := 
sorry

end min_value_symmetry_l1437_143791


namespace probability_first_grade_probability_at_least_one_second_grade_l1437_143720

-- Define conditions
def total_products : ℕ := 10
def first_grade_products : ℕ := 8
def second_grade_products : ℕ := 2
def inspected_products : ℕ := 2
def total_combinations : ℕ := Nat.choose total_products inspected_products
def first_grade_combinations : ℕ := Nat.choose first_grade_products inspected_products
def mixed_combinations : ℕ := first_grade_products * second_grade_products
def second_grade_combinations : ℕ := Nat.choose second_grade_products inspected_products

-- Define probabilities
def P_A : ℚ := first_grade_combinations / total_combinations
def P_B1 : ℚ := mixed_combinations / total_combinations
def P_B2 : ℚ := second_grade_combinations / total_combinations
def P_B : ℚ := P_B1 + P_B2

-- Statements
theorem probability_first_grade : P_A = 28 / 45 := sorry
theorem probability_at_least_one_second_grade : P_B = 17 / 45 := sorry

end probability_first_grade_probability_at_least_one_second_grade_l1437_143720


namespace scientific_notation_l1437_143742

-- Given radius of a water molecule
def radius_of_water_molecule := 0.00000000192

-- Required scientific notation
theorem scientific_notation : radius_of_water_molecule = 1.92 * 10 ^ (-9) :=
by
  sorry

end scientific_notation_l1437_143742


namespace digit_7_occurrences_in_range_20_to_199_l1437_143715

open Set

noncomputable def countDigitOccurrences (low high : ℕ) (digit : ℕ) : ℕ :=
  sorry

theorem digit_7_occurrences_in_range_20_to_199 : 
  countDigitOccurrences 20 199 7 = 38 := 
by
  sorry

end digit_7_occurrences_in_range_20_to_199_l1437_143715


namespace ratio_of_lengths_l1437_143786

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end ratio_of_lengths_l1437_143786


namespace find_a9_for_geo_seq_l1437_143750

noncomputable def geo_seq_a_3_a_13_positive_common_ratio_2 (a_3 a_9 a_13 : ℕ) : Prop :=
  (a_3 * a_13 = 16) ∧ (a_3 > 0) ∧ (a_9 > 0) ∧ (a_13 > 0) ∧ (forall (n₁ n₂ : ℕ), a_9 = a_3 * 2 ^ 6)

theorem find_a9_for_geo_seq (a_3 a_9 a_13 : ℕ) 
  (h : geo_seq_a_3_a_13_positive_common_ratio_2 a_3 a_9 a_13) :
  a_9 = 8 :=
  sorry

end find_a9_for_geo_seq_l1437_143750


namespace rectangle_area_l1437_143739

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 40) : l * b = 75 := by
  sorry

end rectangle_area_l1437_143739


namespace smallest_y_condition_l1437_143738

theorem smallest_y_condition : ∃ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 ∧ y = 167 :=
by 
  sorry

end smallest_y_condition_l1437_143738


namespace hexagon_area_within_rectangle_of_5x4_l1437_143763

-- Define the given conditions
def is_rectangle (length width : ℝ) := length > 0 ∧ width > 0

def vertices_touch_midpoints (length width : ℝ) (hexagon_area : ℝ) : Prop :=
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * (length / 2) * (width / 2)
  let total_triangle_area := 4 * triangle_area
  rectangle_area - total_triangle_area = hexagon_area

-- Formulate the main statement to be proved
theorem hexagon_area_within_rectangle_of_5x4 : 
  vertices_touch_midpoints 5 4 10 := 
by
  -- Proof is omitted for this theorem
  sorry

end hexagon_area_within_rectangle_of_5x4_l1437_143763


namespace bisector_length_is_correct_l1437_143788

noncomputable def length_of_bisector_of_angle_C
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) : ℝ := 3.2

theorem bisector_length_is_correct
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) :
    length_of_bisector_of_angle_C BC AC angleC hBC hAC hAngleC = 3.2 := by
  sorry

end bisector_length_is_correct_l1437_143788


namespace problem_solution_set_l1437_143758

-- Definitions and conditions according to the given problem
def odd_function_domain := {x : ℝ | x ≠ 0}
def function_condition1 (f : ℝ → ℝ) (x : ℝ) : Prop := x > 0 → deriv f x < (3 * f x) / x
def function_condition2 (f : ℝ → ℝ) : Prop := f 1 = 1 / 2
def function_condition3 (f : ℝ → ℝ) : Prop := ∀ x, f (2 * x) = 2 * f x

-- Main proof statement
theorem problem_solution_set (f : ℝ → ℝ)
  (odd_function : ∀ x, f (-x) = -f x)
  (dom : ∀ x, x ∈ odd_function_domain → f x ≠ 0)
  (cond1 : ∀ x, function_condition1 f x)
  (cond2 : function_condition2 f)
  (cond3 : function_condition3 f) :
  {x : ℝ | f x / (4 * x) < 2 * x^2} = {x : ℝ | x < -1 / 4} ∪ {x : ℝ | x > 1 / 4} :=
sorry

end problem_solution_set_l1437_143758


namespace intersection_complement_l1437_143732

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B := {x : ℝ | x < 1}
def complement_B := {x : ℝ | x ≥ 1}

theorem intersection_complement :
  (set_A ∩ complement_B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l1437_143732


namespace least_plates_to_ensure_matching_pair_l1437_143794

theorem least_plates_to_ensure_matching_pair
  (white_plates : ℕ)
  (green_plates : ℕ)
  (red_plates : ℕ)
  (pink_plates : ℕ)
  (purple_plates : ℕ)
  (h_white : white_plates = 2)
  (h_green : green_plates = 6)
  (h_red : red_plates = 8)
  (h_pink : pink_plates = 4)
  (h_purple : purple_plates = 10) :
  ∃ n, n = 6 :=
by
  sorry

end least_plates_to_ensure_matching_pair_l1437_143794


namespace correct_system_l1437_143777

def system_of_equations (x y : ℤ) : Prop :=
  (5 * x + 45 = y) ∧ (7 * x - 3 = y)

theorem correct_system : ∃ x y : ℤ, system_of_equations x y :=
sorry

end correct_system_l1437_143777


namespace total_growing_space_correct_l1437_143774

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end total_growing_space_correct_l1437_143774


namespace percentage_reduction_l1437_143711

-- Define the problem within given conditions
def original_length := 30 -- original length in seconds
def new_length := 21 -- new length in seconds

-- State the theorem that needs to be proved
theorem percentage_reduction (original_length new_length : ℕ) : 
  original_length = 30 → 
  new_length = 21 → 
  ((original_length - new_length) / original_length: ℚ) * 100 = 30 :=
by 
  sorry

end percentage_reduction_l1437_143711


namespace seconds_in_minutes_l1437_143762

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l1437_143762


namespace find_ages_l1437_143755

theorem find_ages (F S : ℕ) (h1 : F + 2 * S = 110) (h2 : 3 * F = 186) :
  F = 62 ∧ S = 24 := by
  sorry

end find_ages_l1437_143755


namespace length_of_bridge_is_correct_l1437_143773

noncomputable def length_of_inclined_bridge (initial_speed : ℕ) (time : ℕ) (acceleration : ℕ) : ℚ :=
  (1 / 60) * (time * initial_speed + (time * (time - 1)) / 2)

theorem length_of_bridge_is_correct : 
  length_of_inclined_bridge 10 18 1 = 5.55 := 
by
  sorry

end length_of_bridge_is_correct_l1437_143773


namespace find_common_difference_l1437_143702

noncomputable def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 = 7 ∧ a 3 + a 6 = 16

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) (h : common_difference a d) : d = 2 :=
by
  sorry

end find_common_difference_l1437_143702


namespace max_value_squared_of_ratio_l1437_143769

-- Definition of positive real numbers with given conditions
variables (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 

-- Main statement
theorem max_value_squared_of_ratio 
  (h_ge : a ≥ b)
  (h_eq_1 : a ^ 2 + y ^ 2 = b ^ 2 + x ^ 2)
  (h_eq_2 : b ^ 2 + x ^ 2 = (a - x) ^ 2 + (b + y) ^ 2)
  (h_range_x : 0 ≤ x ∧ x < a)
  (h_range_y : 0 ≤ y ∧ y < b)
  (h_additional_x : x = a - 2 * b)
  (h_additional_y : y = b / 2) : 
  (a / b) ^ 2 = 4 / 9 := 
sorry

end max_value_squared_of_ratio_l1437_143769


namespace number_of_cases_for_Ds_hearts_l1437_143740

theorem number_of_cases_for_Ds_hearts (hA : 5 ≤ 13) (hB : 4 ≤ 13) (dist : 52 % 4 = 0) : 
  ∃ n, n = 5 ∧ 0 ≤ n ∧ n ≤ 13 := sorry

end number_of_cases_for_Ds_hearts_l1437_143740


namespace scout_weekend_earnings_l1437_143759

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l1437_143759


namespace seq_prime_l1437_143717

/-- A strictly increasing sequence of positive integers. -/
def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

/-- An infinite strictly increasing sequence of positive integers. -/
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n ∧ is_strictly_increasing a

/-- A sequence of distinct primes. -/
def distinct_primes (p : ℕ → ℕ) : Prop :=
  ∀ m n, m ≠ n → p m ≠ p n ∧ Nat.Prime (p n)

/-- The main theorem to be proved. -/
theorem seq_prime (a p : ℕ → ℕ) (h1 : strictly_increasing_sequence a) (h2 : distinct_primes p)
  (h3 : ∀ n, p n ∣ a n) (h4 : ∀ n k, a n - a k = p n - p k) : ∀ n, Nat.Prime (a n) := 
by
  sorry

end seq_prime_l1437_143717


namespace remainder_equality_l1437_143799

theorem remainder_equality (P P' : ℕ) (h1 : P = P' + 10) 
  (h2 : P % 10 = 0) (h3 : P' % 10 = 0) : 
  ((P^2 - P'^2) % 10 = 0) :=
by
  sorry

end remainder_equality_l1437_143799


namespace circle_radius_tangent_to_parabola_l1437_143756

theorem circle_radius_tangent_to_parabola (a : ℝ) (b r : ℝ) :
  (∀ x : ℝ, y = 4 * x ^ 2) ∧ 
  (b = a ^ 2 / 4) ∧ 
  (∀ x : ℝ, x ^ 2 + (4 * x ^ 2 - b) ^ 2 = r ^ 2)  → 
  r = a ^ 2 / 4 := 
  sorry

end circle_radius_tangent_to_parabola_l1437_143756


namespace negation_of_universal_l1437_143764

variable (f : ℝ → ℝ) (m : ℝ)

theorem negation_of_universal :
  (∀ x : ℝ, f x ≥ m) → ¬ (∀ x : ℝ, f x ≥ m) → ∃ x : ℝ, f x < m :=
by
  sorry

end negation_of_universal_l1437_143764


namespace hyperbola_eccentricity_eq_two_l1437_143783

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end hyperbola_eccentricity_eq_two_l1437_143783


namespace polynomials_with_conditions_l1437_143796

theorem polynomials_with_conditions (n : ℕ) (h_pos : 0 < n) :
  (∃ P : Polynomial ℤ, Polynomial.degree P = n ∧ 
      (∃ (k : Fin n → ℤ), Function.Injective k ∧ (∀ i, P.eval (k i) = n) ∧ P.eval 0 = 0)) ↔ 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
sorry

end polynomials_with_conditions_l1437_143796


namespace simplify_expression_l1437_143731

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 20 + 4 * y = 45 * x + 20 + 4 * y :=
by
  sorry

end simplify_expression_l1437_143731


namespace moles_of_NaCl_l1437_143713

def moles_of_reactants (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

theorem moles_of_NaCl (NaCl KNO3 NaNO3 KCl : ℕ) 
  (h : moles_of_reactants NaCl KNO3 NaNO3 KCl) 
  (h2 : KNO3 = 1)
  (h3 : NaNO3 = 1) :
  NaCl = 1 :=
by
  sorry

end moles_of_NaCl_l1437_143713


namespace rotate_and_translate_line_l1437_143766

theorem rotate_and_translate_line :
  let initial_line (x : ℝ) := 3 * x
  let rotated_line (x : ℝ) := - (1 / 3) * x
  let translated_line (x : ℝ) := - (1 / 3) * (x - 1)

  ∀ x : ℝ, translated_line x = - (1 / 3) * x + (1 / 3) := 
by
  intros
  simp
  sorry

end rotate_and_translate_line_l1437_143766


namespace contradiction_assumption_l1437_143728

theorem contradiction_assumption (a b : ℝ) (h : a ≤ 2 ∧ b ≤ 2) : (a > 2 ∨ b > 2) -> false :=
by
  sorry

end contradiction_assumption_l1437_143728
