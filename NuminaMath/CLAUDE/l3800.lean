import Mathlib

namespace greatest_five_digit_with_product_90_l3800_380049

/-- A function that returns true if n is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The greatest five-digit number whose digits have a product of 90 -/
def M : ℕ :=
  sorry

theorem greatest_five_digit_with_product_90 :
  is_five_digit M ∧
  digit_product M = 90 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
  digit_sum M = 18 :=
sorry

end greatest_five_digit_with_product_90_l3800_380049


namespace inequality_proof_l3800_380061

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l3800_380061


namespace modular_congruence_unique_solution_l3800_380004

theorem modular_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 38635 % 23 = n := by
  sorry

end modular_congruence_unique_solution_l3800_380004


namespace product_sum_theorem_l3800_380064

theorem product_sum_theorem (p q r s t : ℤ) :
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 35 := by
sorry

end product_sum_theorem_l3800_380064


namespace new_boy_weight_l3800_380090

theorem new_boy_weight (original_count : ℕ) (original_average : ℝ) (new_average : ℝ) : 
  original_count = 5 →
  original_average = 35 →
  new_average = 36 →
  (original_count + 1) * new_average - original_count * original_average = 41 :=
by
  sorry

end new_boy_weight_l3800_380090


namespace inequality_range_l3800_380086

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) → 
  a ≤ 0 := by
sorry

end inequality_range_l3800_380086


namespace range_a_theorem_l3800_380083

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := a < 1 ∧ a ≠ 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end range_a_theorem_l3800_380083


namespace f_properties_l3800_380053

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end f_properties_l3800_380053


namespace odd_number_product_difference_l3800_380095

theorem odd_number_product_difference (x : ℤ) : 
  Odd x → x * (x + 2) - x * (x - 2) = 44 → x = 11 := by
  sorry

end odd_number_product_difference_l3800_380095


namespace inverse_difference_equals_negative_one_l3800_380056

theorem inverse_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = -1 :=
sorry

end inverse_difference_equals_negative_one_l3800_380056


namespace quadratic_inequality_solution_sets_l3800_380073

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the modified quadratic function
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x - 1) + c - 2 * a * x

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, g a b c x > 0 ↔ 0 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_sets_l3800_380073


namespace correct_division_result_l3800_380024

theorem correct_division_result (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 63)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 42 := by
sorry

end correct_division_result_l3800_380024


namespace same_solution_implies_k_value_l3800_380069

theorem same_solution_implies_k_value (x : ℝ) (k : ℝ) : 
  (2 * x - 1 = 3) ∧ (3 * x + k = 0) → k = -6 := by
  sorry

end same_solution_implies_k_value_l3800_380069


namespace remainder_problem_l3800_380038

theorem remainder_problem (n : ℤ) : 
  ∃ (r : ℕ), r < 25 ∧ 
  n % 25 = r ∧ 
  (n + 15) % 5 = r % 5 → 
  r = 5 := by
sorry

end remainder_problem_l3800_380038


namespace trebled_result_is_69_l3800_380085

theorem trebled_result_is_69 :
  let x : ℕ := 7
  let doubled_plus_nine := 2 * x + 9
  let trebled_result := 3 * doubled_plus_nine
  trebled_result = 69 := by
sorry

end trebled_result_is_69_l3800_380085


namespace simplified_fourth_root_l3800_380005

theorem simplified_fourth_root (a b : ℕ+) :
  (2^9 * 3^5 : ℝ)^(1/4) = (a : ℝ) * ((b : ℝ)^(1/4)) → a + b = 18 := by
  sorry

end simplified_fourth_root_l3800_380005


namespace mans_usual_time_to_office_l3800_380044

/-- Proves that if a man walks at 3/4 of his usual pace and arrives at his office 20 minutes late, 
    his usual time to reach the office is 80 minutes. -/
theorem mans_usual_time_to_office (usual_pace : ℝ) (usual_time : ℝ) 
    (h1 : usual_pace > 0) (h2 : usual_time > 0) : 
    (3 / 4 * usual_pace) * (usual_time + 20) = usual_pace * usual_time → 
    usual_time = 80 := by
  sorry


end mans_usual_time_to_office_l3800_380044


namespace train_speed_l3800_380036

/-- The speed of a train given its length and time to cross a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 7) :
  ∃ (speed : ℝ), abs (speed - 141.43) < 0.01 ∧ speed = (length / time) * 3.6 := by
  sorry


end train_speed_l3800_380036


namespace win_sector_area_l3800_380075

/-- Theorem: Area of WIN sector in a circular spinner game --/
theorem win_sector_area (r : ℝ) (p_win : ℝ) (p_bonus_lose : ℝ) (h1 : r = 8)
  (h2 : p_win = 1 / 4) (h3 : p_bonus_lose = 1 / 8) :
  p_win * π * r^2 = 16 * π := by sorry

end win_sector_area_l3800_380075


namespace solve_equation_one_solve_equation_two_l3800_380098

-- Equation 1
theorem solve_equation_one : 
  {x : ℝ | x^2 - 9 = 0} = {3, -3} := by sorry

-- Equation 2
theorem solve_equation_two :
  {x : ℝ | (x + 1)^3 = -8/27} = {-5/3} := by sorry

end solve_equation_one_solve_equation_two_l3800_380098


namespace cubic_function_zeros_l3800_380006

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  0 < a ∧ a < 2 := by
  sorry

end cubic_function_zeros_l3800_380006


namespace smallest_modulus_w_l3800_380094

theorem smallest_modulus_w (w : ℂ) (h : Complex.abs (w - 8) + Complex.abs (w - 3 * I) = 15) :
  ∃ (w_min : ℂ), Complex.abs w_min ≤ Complex.abs w ∧ Complex.abs w_min = 8 / 5 := by
  sorry

end smallest_modulus_w_l3800_380094


namespace weight_of_new_person_l3800_380030

/-- Theorem: Weight of new person in group weight change scenario -/
theorem weight_of_new_person
  (n : ℕ)  -- Number of persons in the group
  (w : ℝ)  -- Initial total weight of the group
  (r : ℝ)  -- Weight of the person being replaced
  (d : ℝ)  -- Increase in average weight after replacement
  (h1 : n = 10)  -- There are 10 persons
  (h2 : r = 65)  -- The replaced person weighs 65 kg
  (h3 : d = 3.7)  -- The average weight increases by 3.7 kg
  : ∃ x : ℝ, (w - r + x) / n = w / n + d ∧ x = 102 :=
sorry

end weight_of_new_person_l3800_380030


namespace tangent_line_at_x_1_l3800_380088

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ x - y - 1 = 0) ∧
    m = f' 1 ∧
    f 1 = m*1 + b :=
by sorry

end tangent_line_at_x_1_l3800_380088


namespace apple_in_B_l3800_380020

-- Define the boxes
inductive Box
| A
| B
| C

-- Define the location of the apple
def apple_location : Box := Box.B

-- Define the notes on the boxes
def note_A : Prop := apple_location = Box.A
def note_B : Prop := apple_location ≠ Box.B
def note_C : Prop := apple_location ≠ Box.A

-- Define the condition that only one note is true
def only_one_true : Prop :=
  (note_A ∧ ¬note_B ∧ ¬note_C) ∨
  (¬note_A ∧ note_B ∧ ¬note_C) ∨
  (¬note_A ∧ ¬note_B ∧ note_C)

-- Theorem to prove
theorem apple_in_B :
  only_one_true → apple_location = Box.B :=
by sorry

end apple_in_B_l3800_380020


namespace cost_of_one_ring_l3800_380039

/-- The cost of a single ring given the total cost and number of rings. -/
def ring_cost (total_cost : ℕ) (num_rings : ℕ) : ℕ :=
  total_cost / num_rings

/-- Theorem stating that the cost of one ring is $24 given the problem conditions. -/
theorem cost_of_one_ring :
  let total_cost : ℕ := 48
  let num_rings : ℕ := 2
  ring_cost total_cost num_rings = 24 := by
  sorry

end cost_of_one_ring_l3800_380039


namespace line_passes_through_quadrants_l3800_380096

-- Define the type for quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to check if a point (x, y) is in a given quadrant
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.First => x > 0 ∧ y > 0
  | Quadrant.Second => x < 0 ∧ y > 0
  | Quadrant.Third => x < 0 ∧ y < 0
  | Quadrant.Fourth => x > 0 ∧ y < 0

-- Define a function to check if a line passes through a quadrant
def line_passes_through (m b : ℝ) (q : Quadrant) : Prop :=
  ∃ (x y : ℝ), y = m * x + b ∧ in_quadrant x y q

-- State the theorem
theorem line_passes_through_quadrants
  (a b c p : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : (a + b) / c = p)
  (h2 : (b + c) / a = p)
  (h3 : (c + a) / b = p) :
  line_passes_through p p Quadrant.Second ∧
  line_passes_through p p Quadrant.Third :=
sorry

end line_passes_through_quadrants_l3800_380096


namespace lily_bought_20_ducks_l3800_380076

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := sorry

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The total number of birds Lily has -/
def lily_total : ℕ := lily_ducks + lily_geese

/-- The total number of birds Rayden has -/
def rayden_total : ℕ := rayden_ducks + rayden_geese

theorem lily_bought_20_ducks :
  lily_ducks = 20 ∧
  rayden_total = lily_total + 70 := by
  sorry

end lily_bought_20_ducks_l3800_380076


namespace book_arrangement_count_l3800_380062

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let remaining_books := num_math_books + (num_history_books - 2)
  num_history_books * (num_history_books - 1) * Nat.factorial remaining_books

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 5 4 = 60480 := by
  sorry

end book_arrangement_count_l3800_380062


namespace average_pens_sold_per_day_l3800_380093

theorem average_pens_sold_per_day 
  (bundles_sold : ℕ) 
  (days : ℕ) 
  (pens_per_bundle : ℕ) 
  (h1 : bundles_sold = 15) 
  (h2 : days = 5) 
  (h3 : pens_per_bundle = 40) : 
  (bundles_sold * pens_per_bundle) / days = 120 := by
  sorry

end average_pens_sold_per_day_l3800_380093


namespace cubic_function_properties_l3800_380089

-- Define the function f
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Define the derivative of f
def f' (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem cubic_function_properties (b c d : ℝ) :
  (∀ k, (k < 0 ∨ k > 4) → (∃! x, f b c d x = k)) ∧
  (∀ k, (0 < k ∧ k < 4) → (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f b c d x = k ∧ f b c d y = k ∧ f b c d z = k)) →
  (∃ x, f b c d x = 4 ∧ f' b c x = 0) ∧
  (∃ x, f b c d x = 0 ∧ f' b c x = 0) :=
sorry

end cubic_function_properties_l3800_380089


namespace sum_of_solutions_is_zero_l3800_380028

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (6 * x₁ = 150 / x₁) ∧ (6 * x₂ = 150 / x₂) ∧ (x₁ + x₂ = 0) :=
by sorry

end sum_of_solutions_is_zero_l3800_380028


namespace prime_iff_divides_factorial_plus_one_l3800_380025

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) :=
sorry

end prime_iff_divides_factorial_plus_one_l3800_380025


namespace football_field_fertilizer_l3800_380067

/-- Given a football field and fertilizer distribution, calculate the total fertilizer used. -/
theorem football_field_fertilizer 
  (field_area : ℝ) 
  (partial_area : ℝ) 
  (partial_fertilizer : ℝ) 
  (h1 : field_area = 8400)
  (h2 : partial_area = 3500)
  (h3 : partial_fertilizer = 500)
  (h4 : partial_area > 0)
  (h5 : field_area > 0) :
  (field_area * partial_fertilizer) / partial_area = 1200 :=
by sorry

end football_field_fertilizer_l3800_380067


namespace arithmetic_sequence_terms_l3800_380059

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that for an arithmetic sequence with first term 15, 
common difference 4, and last term 159, the number of terms is 37.
-/
theorem arithmetic_sequence_terms (first_term : ℕ) (common_diff : ℕ) (last_term : ℕ) :
  first_term = 15 → common_diff = 4 → last_term = 159 →
  (last_term - first_term) / common_diff + 1 = 37 := by
sorry

end arithmetic_sequence_terms_l3800_380059


namespace engagement_treats_value_l3800_380091

def hotel_nights : ℕ := 2
def hotel_cost_per_night : ℕ := 4000
def car_value : ℕ := 30000

def total_treat_value : ℕ :=
  hotel_nights * hotel_cost_per_night + car_value + 4 * car_value

theorem engagement_treats_value :
  total_treat_value = 158000 := by
  sorry

end engagement_treats_value_l3800_380091


namespace lateral_area_of_specific_prism_l3800_380077

/-- A prism with a square base and a circumscribed sphere -/
structure SquareBasePrism where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Height of the prism -/
  height : ℝ
  /-- Volume of the circumscribed sphere -/
  sphereVolume : ℝ

/-- Theorem: The lateral area of a square-based prism with circumscribed sphere volume 4π/3 and base side length 1 is 4√2 -/
theorem lateral_area_of_specific_prism (p : SquareBasePrism) 
  (h1 : p.baseSide = 1)
  (h2 : p.sphereVolume = 4 * Real.pi / 3) : 
  4 * p.baseSide * p.height = 4 * Real.sqrt 2 := by
  sorry


end lateral_area_of_specific_prism_l3800_380077


namespace birds_in_marsh_end_of_day_l3800_380046

/-- Calculates the total number of birds in the marsh at the end of the day -/
def total_birds_end_of_day (initial_geese initial_ducks geese_departed swans_arrived herons_arrived : ℕ) : ℕ :=
  (initial_geese - geese_departed) + initial_ducks + swans_arrived + herons_arrived

/-- Theorem stating the total number of birds at the end of the day -/
theorem birds_in_marsh_end_of_day :
  total_birds_end_of_day 58 37 15 22 2 = 104 := by
  sorry

end birds_in_marsh_end_of_day_l3800_380046


namespace probability_derek_julia_captains_l3800_380051

theorem probability_derek_julia_captains (total_players : Nat) (num_teams : Nat) (team_size : Nat) (captains_per_team : Nat) :
  total_players = 64 →
  num_teams = 8 →
  team_size = 8 →
  captains_per_team = 2 →
  num_teams * team_size = total_players →
  (probability_both_captains : ℚ) = 5 / 84 :=
by
  sorry

end probability_derek_julia_captains_l3800_380051


namespace inequality_proof_l3800_380017

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end inequality_proof_l3800_380017


namespace line_properties_l3800_380008

/-- The line equation: (a+1)x + y + 2-a = 0 -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Equal intercepts on both coordinate axes -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ line_equation a t 0 ∧ line_equation a 0 t

/-- Line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → ¬(x < 0 ∧ y > 0)

/-- Main theorem -/
theorem line_properties :
  (∀ a : ℝ, equal_intercepts a ↔ (a = 0 ∨ a = 2)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) := by sorry

end line_properties_l3800_380008


namespace difference_of_squares_division_problem_solution_l3800_380045

theorem difference_of_squares_division (a b : ℕ) (h : a > b) : 
  (a ^ 2 - b ^ 2) / (a - b) = a + b := by sorry

theorem problem_solution : (125 ^ 2 - 117 ^ 2) / 8 = 242 := by sorry

end difference_of_squares_division_problem_solution_l3800_380045


namespace inequality_proof_l3800_380041

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
  sorry

end inequality_proof_l3800_380041


namespace part_one_part_two_l3800_380016

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |3 * x + m|

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - 2 * |x - 1|

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 3 ↔ f m x - m ≤ 9) → m = -3 := by sorry

-- Part II
theorem part_two (m : ℝ) (h_m_pos : m > 0) :
  (∃ A B C : ℝ × ℝ, 
    A.2 = 0 ∧ B.2 = 0 ∧ C.2 = g m C.1 ∧
    C.1 ∈ Set.Ioo A.1 B.1 ∧
    (1/2) * |B.1 - A.1| * |C.2| > 60) →
  m > 12 := by sorry

end part_one_part_two_l3800_380016


namespace fruit_arrangement_theorem_l3800_380057

def num_apples : ℕ := 4
def num_oranges : ℕ := 3
def num_bananas : ℕ := 2
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Function to calculate the number of ways to arrange fruits
-- with the constraint that not all apples are consecutive
def arrange_fruits (a o b : ℕ) : ℕ := sorry

theorem fruit_arrangement_theorem :
  arrange_fruits num_apples num_oranges num_bananas = 150 := by sorry

end fruit_arrangement_theorem_l3800_380057


namespace match_probabilities_l3800_380021

/-- A best-of-5 match where the probability of winning each game is 3/5 -/
structure Match :=
  (p : ℝ)
  (h_p : p = 3/5)

/-- The probability of winning 3 consecutive games -/
def prob_win_3_0 (m : Match) : ℝ := m.p^3

/-- The probability of winning the match after losing the first game -/
def prob_win_after_loss (m : Match) : ℝ :=
  m.p^3 + 3 * m.p^3 * (1 - m.p)

/-- The expected number of games played when losing the first game -/
def expected_games_after_loss (m : Match) : ℝ :=
  3 * (1 - m.p)^2 + 4 * (2 * m.p * (1 - m.p)^2 + m.p^3) + 5 * (3 * m.p^2 * (1 - m.p)^2 + m.p^3 * (1 - m.p))

theorem match_probabilities (m : Match) :
  prob_win_3_0 m = 27/125 ∧
  prob_win_after_loss m = 297/625 ∧
  expected_games_after_loss m = 534/125 :=
by sorry

end match_probabilities_l3800_380021


namespace petrol_price_increase_l3800_380023

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h_positive_price : original_price > 0) 
  (h_positive_consumption : original_consumption > 0) : 
  let consumption_reduction := 23.076923076923073 / 100
  let new_consumption := original_consumption * (1 - consumption_reduction)
  let new_price := original_price * original_consumption / new_consumption
  (new_price - original_price) / original_price = 0.3 := by
sorry

end petrol_price_increase_l3800_380023


namespace y_intercept_two_distance_from_origin_one_l3800_380040

-- Define the general equation of line l
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  x + (a + 1) * y + 2 - a = 0

-- Theorem 1: y-intercept is 2
theorem y_intercept_two :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ x - 3 * y + 6 = 0) ∧
  (∃ y : ℝ, line_equation a 0 y ∧ y = 2) :=
sorry

-- Theorem 2: distance from origin is 1
theorem distance_from_origin_one :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ 3 * x + 4 * y + 5 = 0) ∧
  (|2 - a| / Real.sqrt (1 + (a + 1)^2) = 1) :=
sorry

end y_intercept_two_distance_from_origin_one_l3800_380040


namespace grapes_boxes_count_l3800_380033

def asparagus_bundles : ℕ := 60
def asparagus_price : ℚ := 3
def grape_price : ℚ := 2.5
def apple_count : ℕ := 700
def apple_price : ℚ := 0.5
def total_worth : ℚ := 630

theorem grapes_boxes_count :
  ∃ (grape_boxes : ℕ),
    grape_boxes * grape_price +
    asparagus_bundles * asparagus_price +
    apple_count * apple_price = total_worth ∧
    grape_boxes = 40 := by sorry

end grapes_boxes_count_l3800_380033


namespace min_rectangles_to_cover_square_l3800_380058

-- Define the dimensions of the rectangle
def rectangle_width : ℕ := 3
def rectangle_height : ℕ := 4

-- Define the area of the rectangle
def rectangle_area : ℕ := rectangle_width * rectangle_height

-- Define the function to calculate the number of rectangles needed
def rectangles_needed (square_side : ℕ) : ℕ :=
  (square_side * square_side) / rectangle_area

-- Theorem statement
theorem min_rectangles_to_cover_square :
  ∃ (n : ℕ), 
    n > 0 ∧
    rectangles_needed n = 12 ∧
    ∀ (m : ℕ), m > 0 → rectangles_needed m ≥ 12 :=
by sorry

end min_rectangles_to_cover_square_l3800_380058


namespace meals_without_restrictions_l3800_380014

theorem meals_without_restrictions (total_clients : ℕ) (vegan kosher gluten_free halal dairy_free nut_free : ℕ)
  (vegan_kosher vegan_gluten kosher_gluten halal_dairy gluten_nut : ℕ)
  (vegan_halal_gluten kosher_dairy_nut : ℕ)
  (h1 : total_clients = 80)
  (h2 : vegan = 15)
  (h3 : kosher = 18)
  (h4 : gluten_free = 12)
  (h5 : halal = 10)
  (h6 : dairy_free = 8)
  (h7 : nut_free = 4)
  (h8 : vegan_kosher = 5)
  (h9 : vegan_gluten = 6)
  (h10 : kosher_gluten = 3)
  (h11 : halal_dairy = 4)
  (h12 : gluten_nut = 2)
  (h13 : vegan_halal_gluten = 2)
  (h14 : kosher_dairy_nut = 1) :
  total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free - 
    (vegan_kosher + vegan_gluten + kosher_gluten + halal_dairy + gluten_nut) + 
    (vegan_halal_gluten + kosher_dairy_nut)) = 30 := by
  sorry


end meals_without_restrictions_l3800_380014


namespace triangular_sequence_start_fifteenth_triangular_number_l3800_380070

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence of triangular numbers starts with 1, 3, 6, 10, ... -/
theorem triangular_sequence_start :
  [triangular_number 1, triangular_number 2, triangular_number 3, triangular_number 4] = [1, 3, 6, 10] := by sorry

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number :
  triangular_number 15 = 120 := by sorry

end triangular_sequence_start_fifteenth_triangular_number_l3800_380070


namespace log_10_14_in_terms_of_r_and_s_l3800_380078

theorem log_10_14_in_terms_of_r_and_s (r s : ℝ) 
  (h1 : Real.log 2 / Real.log 9 = r) 
  (h2 : Real.log 7 / Real.log 2 = s) : 
  Real.log 14 / Real.log 10 = (s + 1) / (3 + 1 / (2 * r)) := by
  sorry

end log_10_14_in_terms_of_r_and_s_l3800_380078


namespace inequality_holds_iff_k_in_range_l3800_380019

theorem inequality_holds_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
by sorry

end inequality_holds_iff_k_in_range_l3800_380019


namespace sufficient_not_necessary_condition_l3800_380000

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x / y + y / x ≥ 2) ∧
  ¬(∀ x y : ℝ, x / y + y / x ≥ 2 → x > 0 ∧ y > 0) :=
by sorry

end sufficient_not_necessary_condition_l3800_380000


namespace expression_always_positive_l3800_380010

theorem expression_always_positive (a b : ℝ) : a^2 + b^2 + 4*b - 2*a + 6 > 0 := by
  sorry

end expression_always_positive_l3800_380010


namespace a_neither_sufficient_nor_necessary_for_b_l3800_380029

/-- Proposition A: The complex number z satisfies |z-3|+|z+3| is a constant -/
def propositionA (z : ℂ) : Prop :=
  ∃ c : ℝ, ∀ z : ℂ, Complex.abs (z - 3) + Complex.abs (z + 3) = c

/-- Proposition B: The trajectory of the point corresponding to the complex number z in the complex plane is an ellipse -/
def propositionB (z : ℂ) : Prop :=
  ∃ a b : ℝ, ∃ f₁ f₂ : ℂ, ∀ z : ℂ, Complex.abs (z - f₁) + Complex.abs (z - f₂) = a + b

/-- A is neither sufficient nor necessary for B -/
theorem a_neither_sufficient_nor_necessary_for_b :
  (¬∀ z : ℂ, propositionA z → propositionB z) ∧
  (¬∀ z : ℂ, propositionB z → propositionA z) :=
sorry

end a_neither_sufficient_nor_necessary_for_b_l3800_380029


namespace length_AB_line_MN_fixed_point_min_distance_PM_l3800_380063

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Define the tangent points M and N
def tangent_points (P M N : ℝ × ℝ) : Prop :=
  line_l P.1 P.2 ∧
  circle_O M.1 M.2 ∧ circle_O N.1 N.2 ∧
  (P.1 - M.1) * M.1 + (P.2 - M.2) * M.2 = 0 ∧
  (P.1 - N.1) * N.1 + (P.2 - N.2) * N.2 = 0

-- Theorem statements
theorem length_AB : ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := 
sorry

theorem line_MN_fixed_point : ∀ P M N : ℝ × ℝ, tangent_points P M N →
  ∃ t : ℝ, M.1 + t * (N.1 - M.1) = -4/5 ∧ M.2 + t * (N.2 - M.2) = 8/5 :=
sorry

theorem min_distance_PM : ∀ P : ℝ × ℝ, line_l P.1 P.2 →
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    ∀ N : ℝ × ℝ, circle_O N.1 N.2 → 
      Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≤ 
      Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) ∧
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = 1) :=
sorry

end length_AB_line_MN_fixed_point_min_distance_PM_l3800_380063


namespace sum_of_consecutive_integers_l3800_380066

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end sum_of_consecutive_integers_l3800_380066


namespace susan_scores_arithmetic_mean_l3800_380068

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_scores_arithmetic_mean :
  (susan_scores.sum / susan_scores.length : ℝ) = 94 := by
  sorry

end susan_scores_arithmetic_mean_l3800_380068


namespace odd_function_zero_l3800_380074

/-- Definition of an odd function -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Theorem: For an odd function f defined at 0, f(0) = 0 -/
theorem odd_function_zero (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end odd_function_zero_l3800_380074


namespace min_value_on_line_l3800_380027

/-- Given real numbers x and y satisfying the equation x + 2y + 3 = 0,
    the minimum value of √(x² + y² - 2y + 1) is √5. -/
theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 3 = 0) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ ∀ (x' y' : ℝ), x' + 2*y' + 3 = 0 →
    m ≤ Real.sqrt (x'^2 + y'^2 - 2*y' + 1) :=
by sorry

end min_value_on_line_l3800_380027


namespace elle_weekly_practice_hours_l3800_380072

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekday_count : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday practice -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that Elle spends 4 hours practicing piano each week -/
theorem elle_weekly_practice_hours : 
  (weekday_practice * weekday_count + weekday_practice * saturday_factor) / minutes_per_hour = 4 := by
  sorry

end elle_weekly_practice_hours_l3800_380072


namespace quadratic_equation_solution_l3800_380032

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 6
  (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 6) :=
by sorry

end quadratic_equation_solution_l3800_380032


namespace baguette_cost_is_two_l3800_380037

/-- The cost of a single baguette given the initial amount, number of items bought,
    cost of water, and remaining amount after purchase. -/
def baguette_cost (initial_amount : ℚ) (num_baguettes : ℕ) (num_water : ℕ) 
                  (water_cost : ℚ) (remaining_amount : ℚ) : ℚ :=
  (initial_amount - remaining_amount - num_water * water_cost) / num_baguettes

/-- Theorem stating that the cost of each baguette is $2 given the problem conditions. -/
theorem baguette_cost_is_two :
  baguette_cost 50 2 2 1 44 = 2 := by
  sorry

end baguette_cost_is_two_l3800_380037


namespace series_sum_l3800_380034

theorem series_sum : 
  (3/4 : ℚ) + 5/8 + 9/16 + 17/32 + 33/64 + 65/128 - (7/2 : ℚ) = -1/128 := by
  sorry

end series_sum_l3800_380034


namespace union_when_a_is_4_intersection_equals_B_l3800_380080

def A : Set ℝ := {x | x^2 - 5*x - 14 < 0}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3*a - 2}

theorem union_when_a_is_4 : A ∪ B 4 = {x | -2 < x ∧ x ≤ 10} := by sorry

theorem intersection_equals_B (a : ℝ) : A ∩ B a = B a ↔ a < 3 := by sorry

end union_when_a_is_4_intersection_equals_B_l3800_380080


namespace set_operations_l3800_380079

-- Define the sets A and B
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

-- Define the set difference operation
def setDiff (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

theorem set_operations :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧
  (Bᶜ = {x | x ≥ 6 ∨ x ≤ -6}) ∧
  (setDiff A B = {x | x ≥ 6}) ∧
  (setDiff A (setDiff A B) = {x | 4 < x ∧ x < 6}) := by
  sorry

end set_operations_l3800_380079


namespace angle_subtraction_quadrant_l3800_380003

/-- An angle is in the second quadrant if it's between 90° and 180° -/
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

theorem angle_subtraction_quadrant (α : ℝ) (h : is_second_quadrant α) : 
  is_first_quadrant (180 - α) := by
  sorry

end angle_subtraction_quadrant_l3800_380003


namespace classroom_gpa_l3800_380026

theorem classroom_gpa (n : ℝ) (h : n > 0) : 
  (1/3 * n * 45 + 2/3 * n * 60) / n = 55 := by
  sorry

end classroom_gpa_l3800_380026


namespace coach_a_basketballs_l3800_380043

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The cost of each basketball in dollars -/
def basketball_cost : ℚ := 29

/-- The total cost of Coach B's purchases in dollars -/
def coach_b_cost : ℚ := 14 * 2.5 + 18

/-- The difference in cost between Coach A and Coach B's purchases in dollars -/
def cost_difference : ℚ := 237

theorem coach_a_basketballs :
  basketball_cost * num_basketballs = coach_b_cost + cost_difference := by
  sorry


end coach_a_basketballs_l3800_380043


namespace eighteen_power_mnp_l3800_380012

theorem eighteen_power_mnp (m n p : ℕ) (P Q R : ℕ) 
  (hP : P = 2^m) (hQ : Q = 3^n) (hR : R = 5^p) :
  18^(m*n*p) = P^(n*p) * Q^(2*m*p) := by
  sorry

end eighteen_power_mnp_l3800_380012


namespace books_borrowed_after_lunch_l3800_380082

theorem books_borrowed_after_lunch (initial_books : ℕ) (borrowed_by_lunch : ℕ) (added_after_lunch : ℕ) (remaining_by_evening : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  added_after_lunch = 40 →
  remaining_by_evening = 60 →
  initial_books - borrowed_by_lunch + added_after_lunch - remaining_by_evening = 30 := by
sorry

end books_borrowed_after_lunch_l3800_380082


namespace last_three_average_l3800_380013

theorem last_three_average (list : List ℝ) (h1 : list.length = 7) 
  (h2 : list.sum / 7 = 60) (h3 : (list.take 4).sum / 4 = 55) : 
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end last_three_average_l3800_380013


namespace quadratic_roots_existence_l3800_380009

theorem quadratic_roots_existence : 
  (∃ x : ℝ, x^2 + x = 0) ∧ 
  (∃ x : ℝ, 5*x^2 - 4*x - 1 = 0) ∧ 
  (∃ x : ℝ, 3*x^2 - 4*x + 1 = 0) ∧ 
  (∀ x : ℝ, 4*x^2 - 5*x + 2 ≠ 0) := by
sorry

end quadratic_roots_existence_l3800_380009


namespace total_investment_l3800_380035

/-- Proves that the total investment of Vishal, Trishul, and Raghu is 5780 Rs. -/
theorem total_investment (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2000)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = raghu_investment * 0.9 * 1.1) :
  raghu_investment + raghu_investment * 0.9 + raghu_investment * 0.9 * 1.1 = 5780 :=
by sorry


end total_investment_l3800_380035


namespace equation_solution_l3800_380001

theorem equation_solution : ∃ x : ℚ, (1 / 7 + 7 / x = 16 / x + 1 / 16) ∧ x = 112 := by
  sorry

end equation_solution_l3800_380001


namespace melody_reading_fraction_l3800_380081

theorem melody_reading_fraction (english : ℕ) (science : ℕ) (civics : ℕ) (chinese : ℕ) 
  (total_pages : ℕ) (h1 : english = 20) (h2 : science = 16) (h3 : civics = 8) (h4 : chinese = 12) 
  (h5 : total_pages = 14) :
  ∃ (f : ℚ), f * (english + science + civics + chinese : ℚ) = total_pages ∧ f = 1/4 := by
sorry

end melody_reading_fraction_l3800_380081


namespace difference_of_squares_example_l3800_380047

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end difference_of_squares_example_l3800_380047


namespace five_letter_words_count_l3800_380022

def alphabet_size : ℕ := 26
def vowel_count : ℕ := 5

theorem five_letter_words_count : 
  (alphabet_size^3 * vowel_count : ℕ) = 87880 := by
sorry

end five_letter_words_count_l3800_380022


namespace book_price_increase_l3800_380097

/-- Calculates the new price of a book after a percentage increase -/
theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  original_price = 300 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = 390 := by
sorry

end book_price_increase_l3800_380097


namespace curve_line_tangent_l3800_380015

/-- The curve y = √(4 - x²) and the line y = m have exactly one common point if and only if m = 2 -/
theorem curve_line_tangent (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = Real.sqrt (4 - p.1^2) ∧ p.2 = m) ↔ m = 2 := by
  sorry

end curve_line_tangent_l3800_380015


namespace newer_train_distance_calculation_l3800_380092

/-- The distance traveled by the older train in miles -/
def older_train_distance : ℝ := 300

/-- The percentage increase in distance for the newer train -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the newer train in miles -/
def newer_train_distance : ℝ := older_train_distance * (1 + percentage_increase)

theorem newer_train_distance_calculation : newer_train_distance = 390 := by
  sorry

end newer_train_distance_calculation_l3800_380092


namespace f_composition_equals_126_l3800_380007

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 4

-- State the theorem
theorem f_composition_equals_126 : f (f (f 2)) = 126 := by sorry

end f_composition_equals_126_l3800_380007


namespace parabola_point_distance_l3800_380099

/-- For a parabola y = ax², if a point P(x₀, 2) on the parabola is at a distance of 3 
    from the focus, then the distance from P to the y-axis is 2√2. -/
theorem parabola_point_distance (a : ℝ) (x₀ : ℝ) :
  (2 = a * x₀^2) →                          -- P is on the parabola
  ((x₀ - 0)^2 + (2 - 1/(4*a))^2 = 3^2) →    -- Distance from P to focus is 3
  |x₀| = 2 * Real.sqrt 2 :=                 -- Distance from P to y-axis is 2√2
by sorry

end parabola_point_distance_l3800_380099


namespace hypotenuse_length_l3800_380042

theorem hypotenuse_length (x y : ℝ) : 
  2 * x^2 - 8 * x + 7 = 0 →
  2 * y^2 - 8 * y + 7 = 0 →
  x ≠ y →
  x > 0 →
  y > 0 →
  x^2 + y^2 = 3^2 :=
by sorry

end hypotenuse_length_l3800_380042


namespace initial_to_doubled_ratio_l3800_380018

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 5) = 105 → x / (2 * x) = 1 / 2 := by
  sorry

end initial_to_doubled_ratio_l3800_380018


namespace inverse_fraction_minus_abs_diff_l3800_380087

theorem inverse_fraction_minus_abs_diff : (1/3)⁻¹ - |Real.sqrt 3 - 3| = Real.sqrt 3 := by
  sorry

end inverse_fraction_minus_abs_diff_l3800_380087


namespace smallestPalindromeNumber_satisfies_conditions_l3800_380011

/-- A function to check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  (n.digits base).reverse = n.digits base

/-- The smallest positive integer greater than 10 that is a palindrome in base 2 and 4, and is odd -/
def smallestPalindromeNumber : ℕ := 17

/-- Theorem stating that smallestPalindromeNumber satisfies all conditions -/
theorem smallestPalindromeNumber_satisfies_conditions :
  smallestPalindromeNumber > 10 ∧
  isPalindrome smallestPalindromeNumber 2 ∧
  isPalindrome smallestPalindromeNumber 4 ∧
  Odd smallestPalindromeNumber ∧
  ∀ n : ℕ, n > 10 → isPalindrome n 2 → isPalindrome n 4 → Odd n →
    n ≥ smallestPalindromeNumber :=
by sorry

#eval smallestPalindromeNumber

end smallestPalindromeNumber_satisfies_conditions_l3800_380011


namespace range_of_a_l3800_380048

theorem range_of_a (a : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) :=
by sorry

end range_of_a_l3800_380048


namespace equation_solution_l3800_380071

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 15))) = 54 ∧ x = 15 := by
  sorry

end equation_solution_l3800_380071


namespace pauls_crayons_l3800_380031

/-- Given Paul's crayon situation, prove the difference between given and lost crayons. -/
theorem pauls_crayons (initial : ℕ) (given : ℕ) (lost : ℕ) 
  (h1 : initial = 589) 
  (h2 : given = 571) 
  (h3 : lost = 161) : 
  given - lost = 410 := by
  sorry

end pauls_crayons_l3800_380031


namespace circle_center_and_radius_l3800_380054

/-- Given a circle with equation x^2 + y^2 - 2x + 6y = 0, 
    prove that its center is at (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = Real.sqrt 10 ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ↔ 
                   p.1^2 + p.2^2 - 2*p.1 + 6*p.2 = 0 :=
sorry

end circle_center_and_radius_l3800_380054


namespace mr_zhang_birthday_l3800_380084

-- Define the possible dates
inductive Date
| feb5 | feb7 | feb9
| may5 | may8
| aug4 | aug7
| sep4 | sep6 | sep9

def Date.month : Date → Nat
| .feb5 | .feb7 | .feb9 => 2
| .may5 | .may8 => 5
| .aug4 | .aug7 => 8
| .sep4 | .sep6 | .sep9 => 9

def Date.day : Date → Nat
| .feb5 => 5
| .feb7 => 7
| .feb9 => 9
| .may5 => 5
| .may8 => 8
| .aug4 => 4
| .aug7 => 7
| .sep4 => 4
| .sep6 => 6
| .sep9 => 9

-- Define the statements made by A and B
def A_statement1 (d : Date) : Prop := 
  ∃ d' : Date, d.month = d'.month ∧ d ≠ d'

def B_statement (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' → d.day ≠ d'.day

def A_statement2 (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' ∧ B_statement d' → d = d'

-- Theorem to prove
theorem mr_zhang_birthday : 
  ∃! d : Date, A_statement1 d ∧ B_statement d ∧ A_statement2 d ∧ d = Date.aug4 := by
  sorry

end mr_zhang_birthday_l3800_380084


namespace savings_percentage_is_twenty_percent_l3800_380052

def monthly_salary : ℝ := 6250
def savings_after_increase : ℝ := 250
def expense_increase_rate : ℝ := 0.2

theorem savings_percentage_is_twenty_percent :
  ∃ P : ℝ, 
    savings_after_increase = monthly_salary - (1 + expense_increase_rate) * (monthly_salary - (P / 100) * monthly_salary) ∧
    P = 20 := by
  sorry

end savings_percentage_is_twenty_percent_l3800_380052


namespace closure_of_A_range_of_a_l3800_380065

-- Define set A
def A : Set ℝ := {x | x < -1 ∨ x > -1/2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for the closure of A
theorem closure_of_A : 
  closure A = {x : ℝ | -1 ≤ x ∧ x ≤ -1/2} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  (∃ a : ℝ, A ∪ B a = Set.univ) ↔ ∃ a : ℝ, -3/2 ≤ a ∧ a ≤ 0 := by sorry

end closure_of_A_range_of_a_l3800_380065


namespace metallic_sheet_length_l3800_380002

/-- Given a rectangular metallic sheet with width 36 m, from which squares of 8 m are cut from each corner
    to form a box with volume 5120 m³, prove that the length of the original sheet is 48 m. -/
theorem metallic_sheet_length (L : ℝ) : 
  let W : ℝ := 36
  let cut_length : ℝ := 8
  let box_volume : ℝ := 5120
  (L - 2 * cut_length) * (W - 2 * cut_length) * cut_length = box_volume →
  L = 48 := by
sorry

end metallic_sheet_length_l3800_380002


namespace total_spending_l3800_380050

/-- Represents the amount spent by Ben -/
def ben_spent : ℝ := sorry

/-- Represents the amount spent by David -/
def david_spent : ℝ := sorry

/-- Ben spends $1 for every $0.75 David spends -/
axiom spending_ratio : david_spent = 0.75 * ben_spent

/-- Ben spends $12.50 more than David -/
axiom spending_difference : ben_spent = david_spent + 12.50

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

/-- Theorem: The total amount spent by Ben and David is $87.50 -/
theorem total_spending : total_spent = 87.50 := by sorry

end total_spending_l3800_380050


namespace intersection_of_A_and_B_l3800_380055

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l3800_380055


namespace kyle_to_grant_ratio_l3800_380060

def parker_distance : ℝ := 16

def grant_distance : ℝ := parker_distance * 1.25

def kyle_distance : ℝ := parker_distance + 24

theorem kyle_to_grant_ratio : kyle_distance / grant_distance = 2 := by
  sorry

end kyle_to_grant_ratio_l3800_380060
