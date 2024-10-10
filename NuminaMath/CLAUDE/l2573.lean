import Mathlib

namespace no_factors_l2573_257391

def main_polynomial (z : ℂ) : ℂ := z^6 + 3*z^3 + 18

def option1 (z : ℂ) : ℂ := z^3 + 6
def option2 (z : ℂ) : ℂ := z - 2
def option3 (z : ℂ) : ℂ := z^3 - 6
def option4 (z : ℂ) : ℂ := z^3 - 3*z - 9

theorem no_factors :
  (∀ z, main_polynomial z ≠ 0 → option1 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option2 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option3 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option4 z ≠ 0) :=
by sorry

end no_factors_l2573_257391


namespace no_cube_in_sequence_l2573_257381

theorem no_cube_in_sequence : ∀ (n : ℕ), ¬ ∃ (k : ℤ), 2^(2^n) + 1 = k^3 := by sorry

end no_cube_in_sequence_l2573_257381


namespace base_b_sum_equals_21_l2573_257327

-- Define the sum of single-digit numbers in base b
def sum_single_digits (b : ℕ) : ℕ := (b * (b - 1)) / 2

-- Define the value 21 in base b
def value_21_base_b (b : ℕ) : ℕ := 2 * b + 1

-- Theorem statement
theorem base_b_sum_equals_21 :
  ∃ b : ℕ, b > 1 ∧ sum_single_digits b = value_21_base_b b ∧ b = 7 :=
sorry

end base_b_sum_equals_21_l2573_257327


namespace count_subset_pairs_formula_l2573_257311

/-- The number of pairs of non-empty subsets (A, B) of {1, 2, ..., n} such that
    the maximum element of A is less than the minimum element of B -/
def count_subset_pairs (n : ℕ) : ℕ :=
  (n - 2) * 2^(n - 1) + 1

/-- Theorem stating that for any integer n ≥ 3, the count of subset pairs
    satisfying the given condition is equal to (n-2) * 2^(n-1) + 1 -/
theorem count_subset_pairs_formula (n : ℕ) (h : n ≥ 3) :
  count_subset_pairs n = (n - 2) * 2^(n - 1) + 1 := by
  sorry

end count_subset_pairs_formula_l2573_257311


namespace wall_area_calculation_l2573_257351

theorem wall_area_calculation (regular_area : ℝ) (jumbo_ratio : ℝ) (length_ratio : ℝ) :
  regular_area = 70 →
  jumbo_ratio = 1 / 3 →
  length_ratio = 3 →
  (regular_area + jumbo_ratio / (1 - jumbo_ratio) * regular_area * length_ratio) = 175 :=
by
  sorry

end wall_area_calculation_l2573_257351


namespace bus_speed_excluding_stoppages_l2573_257365

/-- Given a bus that stops for 24 minutes per hour and has a speed of 45 kmph including stoppages,
    its speed excluding stoppages is 75 kmph. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stops : ℝ) 
  (h1 : stop_time = 24)
  (h2 : speed_with_stops = 45) : 
  speed_with_stops * (60 / (60 - stop_time)) = 75 := by
  sorry


end bus_speed_excluding_stoppages_l2573_257365


namespace medicine_survey_l2573_257342

theorem medicine_survey (total : ℕ) (cold : ℕ) (stomach : ℕ) 
  (h_total : total = 100)
  (h_cold : cold = 75)
  (h_stomach : stomach = 80)
  (h_cold_le_total : cold ≤ total)
  (h_stomach_le_total : stomach ≤ total) :
  ∃ (max_both min_both : ℕ),
    max_both ≤ cold ∧
    max_both ≤ stomach ∧
    cold + stomach - max_both ≤ total ∧
    max_both = 75 ∧
    min_both ≥ 0 ∧
    min_both ≤ cold ∧
    min_both ≤ stomach ∧
    cold + stomach - min_both ≥ total ∧
    min_both = 55 := by
  sorry

end medicine_survey_l2573_257342


namespace total_interest_calculation_l2573_257370

/-- Calculates the total interest for a loan split into two parts with different interest rates -/
theorem total_interest_calculation 
  (A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : A + B = 10000) : 
  ∃ I : ℝ, I = 0.08 * A + 0.1 * B := by
  sorry

#check total_interest_calculation

end total_interest_calculation_l2573_257370


namespace sum_of_squares_l2573_257386

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 12) : x^2 + y^2 = 460 := by
  sorry

end sum_of_squares_l2573_257386


namespace softball_team_ratio_l2573_257334

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
    women = men + 6 → 
    men + women = 16 → 
    (men : ℚ) / women = 5 / 11 := by
  sorry

end softball_team_ratio_l2573_257334


namespace fraction_value_l2573_257389

theorem fraction_value : 
  (10 + (-9) + 8 + (-7) + 6 + (-5) + 4 + (-3) + 2 + (-1)) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1/2 := by
  sorry

end fraction_value_l2573_257389


namespace canoe_upstream_speed_l2573_257361

/-- Given a canoe with a speed in still water and a downstream speed, calculate its upstream speed -/
theorem canoe_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 12.5)
  (h2 : speed_downstream = 16) :
  speed_still - (speed_downstream - speed_still) = 9 := by
  sorry

#check canoe_upstream_speed

end canoe_upstream_speed_l2573_257361


namespace cube_square_third_smallest_prime_l2573_257323

/-- The third smallest prime number -/
def third_smallest_prime : Nat := 5

/-- The cube of the square of the third smallest prime number -/
def result : Nat := (third_smallest_prime ^ 2) ^ 3

theorem cube_square_third_smallest_prime :
  result = 15625 := by sorry

end cube_square_third_smallest_prime_l2573_257323


namespace polynomial_value_range_l2573_257322

/-- A polynomial with integer coefficients that equals 5 for five different integer inputs -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a b c d e : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5

theorem polynomial_value_range (P : ℤ → ℤ) (h : IntPolynomial P) :
  ¬∃ x : ℤ, ((-6 : ℤ) ≤ P x ∧ P x ≤ 4) ∨ (6 ≤ P x ∧ P x ≤ 16) := by
  sorry

end polynomial_value_range_l2573_257322


namespace custom_mult_four_three_l2573_257335

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) := 2 * a^2 + 3 * b - a * b

/-- Theorem stating that 4 * 3 = 29 under the custom multiplication -/
theorem custom_mult_four_three : custom_mult 4 3 = 29 := by
  sorry

end custom_mult_four_three_l2573_257335


namespace unique_c_l2573_257392

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x - 12

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x < 0 ↔ (x < 2 ∨ x > 7)

-- Theorem statement
theorem unique_c : ∃! c : ℝ, condition c :=
  sorry

end unique_c_l2573_257392


namespace square_area_with_four_circles_l2573_257305

/-- The area of a square containing four circles of radius 7 inches, 
    arranged so that two circles fit into the width and height of the square. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end square_area_with_four_circles_l2573_257305


namespace dinner_bill_proof_l2573_257393

theorem dinner_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * (total / n + extra) = total →
  total = 270 := by
sorry

end dinner_bill_proof_l2573_257393


namespace cuboid_edge_length_l2573_257337

/-- The surface area of a cuboid given its three edge lengths -/
def cuboidSurfaceArea (x y z : ℝ) : ℝ := 2 * (x * y + x * z + y * z)

/-- Theorem stating that if a cuboid with edges x, 5, and 6 has surface area 148, then x = 4 -/
theorem cuboid_edge_length (x : ℝ) :
  cuboidSurfaceArea x 5 6 = 148 → x = 4 := by
  sorry

end cuboid_edge_length_l2573_257337


namespace ball_probability_l2573_257329

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : purple = 1)
  (h7 : total = white + green + yellow + red + purple) :
  (total - (red + purple)) / total = 14 / 15 := by
  sorry

end ball_probability_l2573_257329


namespace fraction_addition_l2573_257362

theorem fraction_addition : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end fraction_addition_l2573_257362


namespace union_implies_m_equals_two_l2573_257353

theorem union_implies_m_equals_two (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end union_implies_m_equals_two_l2573_257353


namespace parabola_b_value_l2573_257354

/-- Given a parabola y = ax^2 + bx + c with vertex (p, -p) and passing through (0, p),
    where p ≠ 0, the value of b is -4/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 / p := by sorry

end parabola_b_value_l2573_257354


namespace find_d_l2573_257376

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x^2 + 5*x - 36 = 0 ∧ x = ⌊d⌋) ∧ 
  (∃ y : ℝ, 3*y^2 - 11*y + 2 = 0 ∧ y = d - ⌊d⌋) ∧
  d = 13/3 := by
sorry

end find_d_l2573_257376


namespace sequence_sum_formula_l2573_257368

/-- Given a sequence of positive real numbers {aₙ} where the sum of the first n terms
    Sₙ satisfies Sₙ = (1/2)(aₙ + 1/aₙ), prove that aₙ = √n - √(n-1) for all positive integers n. -/
theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h_pos : ∀ k, k > 0 → a k > 0)
  (h_sum : ∀ k, k > 0 → S k = (1/2) * (a k + 1 / a k)) :
  a n = Real.sqrt n - Real.sqrt (n - 1) :=
by sorry

end sequence_sum_formula_l2573_257368


namespace integral_f_equals_four_thirds_l2573_257373

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x < Real.exp 1 then 1/x
  else 0  -- undefined elsewhere

-- State the theorem
theorem integral_f_equals_four_thirds :
  ∫ x in (0)..(Real.exp 1), f x = 4/3 := by sorry

end integral_f_equals_four_thirds_l2573_257373


namespace equation_solution_l2573_257352

theorem equation_solution : ∃! x : ℚ, (4 * x - 12) / 3 = (3 * x + 6) / 5 ∧ x = 78 / 11 := by
  sorry

end equation_solution_l2573_257352


namespace zero_exponent_equals_one_l2573_257315

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end zero_exponent_equals_one_l2573_257315


namespace dad_steps_l2573_257348

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  3 * s.masha = 5 * s.dad

/-- The ratio of steps between Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  3 * s.yasha = 5 * s.masha

/-- The total number of steps taken by Masha and Yasha -/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : masha_yasha_total s) :
  s.dad = 90 := by
  sorry


end dad_steps_l2573_257348


namespace unique_factorization_1386_l2573_257338

/-- Two-digit numbers are natural numbers between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A factorization of 1386 into two two-digit numbers -/
structure Factorization :=
  (a b : ℕ)
  (h1 : TwoDigitNumber a)
  (h2 : TwoDigitNumber b)
  (h3 : a * b = 1386)

/-- Two factorizations are considered the same if they have the same factors (in any order) -/
def Factorization.equiv (f g : Factorization) : Prop :=
  (f.a = g.a ∧ f.b = g.b) ∨ (f.a = g.b ∧ f.b = g.a)

/-- The main theorem stating that there is exactly one factorization of 1386 into two-digit numbers -/
theorem unique_factorization_1386 : 
  ∃! (f : Factorization), True :=
sorry

end unique_factorization_1386_l2573_257338


namespace average_of_multiples_l2573_257356

theorem average_of_multiples (x : ℝ) : 
  let terms := [0, 2*x, 4*x, 8*x, 16*x]
  let multiplied_terms := List.map (· * 3) terms
  List.sum multiplied_terms / 5 = 18 * x := by
sorry

end average_of_multiples_l2573_257356


namespace smaller_number_proof_l2573_257341

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end smaller_number_proof_l2573_257341


namespace cost_of_750_candies_l2573_257312

/-- The cost of buying a specific number of chocolate candies given the following conditions:
  * A box contains a fixed number of candies
  * A box costs a fixed amount
  * There is a discount percentage for buying more than a certain number of boxes
  * We need to buy a specific number of candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (discount_percentage : ℚ) 
  (discount_threshold : ℕ) (total_candies : ℕ) : ℚ :=
  let boxes_needed : ℕ := (total_candies + candies_per_box - 1) / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  if boxes_needed > discount_threshold
  then total_cost * (1 - discount_percentage)
  else total_cost

theorem cost_of_750_candies :
  cost_of_candies 30 (7.5) (1/10) 20 750 = (168.75) := by
  sorry

end cost_of_750_candies_l2573_257312


namespace impossible_distance_l2573_257374

/-- Two circles with no common points -/
structure DisjointCircles where
  r₁ : ℝ
  r₂ : ℝ
  d : ℝ
  h₁ : r₁ = 2
  h₂ : r₂ = 5
  h₃ : d < r₂ - r₁ ∨ d > r₂ + r₁

theorem impossible_distance (c : DisjointCircles) : c.d ≠ 5 := by
  sorry

end impossible_distance_l2573_257374


namespace pickle_discount_l2573_257328

/-- Calculates the discount on a jar of pickles based on grocery purchases and change received --/
theorem pickle_discount (meat_price meat_weight buns_price lettuce_price tomato_price tomato_weight pickle_price bill change : ℝ) :
  meat_price = 3.5 ∧
  meat_weight = 2 ∧
  buns_price = 1.5 ∧
  lettuce_price = 1 ∧
  tomato_price = 2 ∧
  tomato_weight = 1.5 ∧
  pickle_price = 2.5 ∧
  bill = 20 ∧
  change = 6 →
  pickle_price - ((meat_price * meat_weight + buns_price + lettuce_price + tomato_price * tomato_weight + pickle_price) - (bill - change)) = 1 := by
  sorry

end pickle_discount_l2573_257328


namespace all_star_arrangement_l2573_257310

def number_of_arrangements (n_cubs : ℕ) (n_red_sox : ℕ) (n_yankees : ℕ) : ℕ :=
  let n_cubs_with_coach := n_cubs + 1
  let n_teams := 3
  n_teams.factorial * n_cubs_with_coach.factorial * n_red_sox.factorial * n_yankees.factorial

theorem all_star_arrangement :
  number_of_arrangements 4 3 2 = 8640 := by
  sorry

end all_star_arrangement_l2573_257310


namespace functional_equation_solution_l2573_257300

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = x + f (f y)

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + c for some real constant c -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by
  sorry

end functional_equation_solution_l2573_257300


namespace soccer_league_teams_l2573_257301

theorem soccer_league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end soccer_league_teams_l2573_257301


namespace prime_divides_abc_l2573_257360

theorem prime_divides_abc (p a b c : ℤ) (hp : Prime p)
  (h1 : (6 : ℤ) ∣ p + 1)
  (h2 : p ∣ a + b + c)
  (h3 : p ∣ a^4 + b^4 + c^4) :
  p ∣ a ∧ p ∣ b ∧ p ∣ c := by
  sorry

end prime_divides_abc_l2573_257360


namespace product_real_condition_l2573_257395

theorem product_real_condition (a b c d : ℝ) :
  (∃ (x : ℝ), (a + b * Complex.I) * (c + d * Complex.I) = x) ↔ a * d + b * c = 0 := by
  sorry

end product_real_condition_l2573_257395


namespace product_equals_sum_solutions_l2573_257385

theorem product_equals_sum_solutions (a b c d e f g : ℕ+) :
  a * b * c * d * e * f * g = a + b + c + d + e + f + g →
  ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 3 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 7 ∧ g = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 4 ∧ g = 3) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 3 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 3 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 4 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 3) ∨
   (a = 7 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 2)) := by
  sorry

end product_equals_sum_solutions_l2573_257385


namespace marble_probability_l2573_257367

theorem marble_probability (total_marbles : ℕ) (p_white p_green p_yellow : ℚ) :
  total_marbles = 250 →
  p_white = 2 / 5 →
  p_green = 1 / 4 →
  p_yellow = 1 / 10 →
  1 - (p_white + p_green + p_yellow) = 1 / 4 := by
  sorry

end marble_probability_l2573_257367


namespace min_moves_to_exit_l2573_257309

/-- Represents the direction of car movement -/
inductive Direction
| Left
| Right
| Up
| Down

/-- Represents a car in the parking lot -/
structure Car where
  id : Nat
  position : Nat × Nat

/-- Represents the parking lot -/
structure ParkingLot where
  cars : List Car
  width : Nat
  height : Nat

/-- Represents a move in the solution -/
structure Move where
  car : Car
  direction : Direction

/-- Checks if a car can exit the parking lot -/
def canExit (pl : ParkingLot) (car : Car) : Prop :=
  sorry

/-- Checks if a sequence of moves is valid -/
def isValidMoveSequence (pl : ParkingLot) (moves : List Move) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_exit (pl : ParkingLot) (car : Car) :
  (∃ (moves : List Move), isValidMoveSequence pl moves ∧ canExit pl car) →
  (∃ (minMoves : List Move), isValidMoveSequence pl minMoves ∧ canExit pl car ∧ minMoves.length = 6) :=
sorry

end min_moves_to_exit_l2573_257309


namespace sum_of_x_and_y_l2573_257330

theorem sum_of_x_and_y (x y : ℤ) : x - y = 200 → y = 235 → x + y = 670 := by
  sorry

end sum_of_x_and_y_l2573_257330


namespace train_length_l2573_257358

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 235 →
  (train_speed * crossing_time) - bridge_length = 140 := by
sorry

end train_length_l2573_257358


namespace pink_cubes_count_l2573_257397

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a colored cube with a given side length and number of colored faces -/
structure ColoredCube extends Cube where
  coloredFaces : ℕ

/-- Calculates the number of smaller cubes with color when a large cube is cut -/
def coloredCubesCount (largeCube : Cube) (coloredFaces : ℕ) : ℕ :=
  sorry

theorem pink_cubes_count :
  let largeCube : Cube := ⟨125⟩
  let coloredFaces : ℕ := 2
  coloredCubesCount largeCube coloredFaces = 46 := by
  sorry

end pink_cubes_count_l2573_257397


namespace class_photo_cost_l2573_257319

/-- The total cost of class photos for a given number of students -/
def total_cost (students : ℕ) (fixed_price : ℚ) (fixed_photos : ℕ) (additional_cost : ℚ) : ℚ :=
  fixed_price + (additional_cost * (students - fixed_photos))

/-- Proof that the total cost for the class photo is 139.5 yuan -/
theorem class_photo_cost :
  let students : ℕ := 54
  let fixed_price : ℚ := 24.5
  let fixed_photos : ℕ := 4
  let additional_cost : ℚ := 2.3
  total_cost students fixed_price fixed_photos additional_cost = 139.5 := by
  sorry

end class_photo_cost_l2573_257319


namespace mixed_number_calculation_l2573_257325

theorem mixed_number_calculation : 
  25 * ((5 + 2/7) - (3 + 3/5)) / ((3 + 1/6) + (2 + 1/4)) = 7 + 49/91 := by
  sorry

end mixed_number_calculation_l2573_257325


namespace shoe_discount_ratio_l2573_257388

/-- Proves the ratio of extra discount to total amount before discount is 1:4 --/
theorem shoe_discount_ratio :
  let first_pair_price : ℚ := 40
  let second_pair_price : ℚ := 60
  let discount_rate : ℚ := 1/2
  let total_paid : ℚ := 60
  let cheaper_pair_price := min first_pair_price second_pair_price
  let discount_amount := discount_rate * cheaper_pair_price
  let total_before_extra_discount := first_pair_price + second_pair_price - discount_amount
  let extra_discount := total_before_extra_discount - total_paid
  extra_discount / total_before_extra_discount = 1/4 := by
sorry

end shoe_discount_ratio_l2573_257388


namespace simplify_expression_1_simplify_expression_2_l2573_257384

-- Define variables
variable (x y a b : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := by
  sorry

end simplify_expression_1_simplify_expression_2_l2573_257384


namespace xyz_inequality_l2573_257364

theorem xyz_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h_eq : x^2 + y^2 + z^2 + x*y*z = 4) : 
  x*y*z ≤ x*y + y*z + z*x ∧ x*y + y*z + z*x ≤ x*y*z + 2 := by
  sorry

end xyz_inequality_l2573_257364


namespace interest_rate_is_ten_percent_l2573_257375

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ :=
  principal * time * rate

/-- Given conditions -/
def principal : ℝ := 2500
def time : ℝ := 4
def interest : ℝ := 1000

/-- Theorem to prove -/
theorem interest_rate_is_ten_percent :
  ∃ (rate : ℝ), simple_interest principal time rate = interest ∧ rate = 0.1 := by
  sorry

end interest_rate_is_ten_percent_l2573_257375


namespace tan_theta_in_terms_of_x_y_l2573_257399

theorem tan_theta_in_terms_of_x_y (θ x y : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sin (θ/2) = Real.sqrt ((y - x)/(y + x))) : 
  Real.tan θ = (2 * Real.sqrt (x * y)) / (3 * x - y) := by
sorry

end tan_theta_in_terms_of_x_y_l2573_257399


namespace smallest_x_value_l2573_257349

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (215 + x)) : 
  ∀ z : ℕ+, z < x → (3 : ℚ) / 4 ≠ y / (215 + z) :=
sorry

end smallest_x_value_l2573_257349


namespace find_number_l2573_257350

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.4 * 120 + 180 ∧ x = 456 := by
  sorry

end find_number_l2573_257350


namespace boys_at_reunion_l2573_257313

/-- The number of handshakes between n boys, where each boy shakes hands
    exactly once with each of the others. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 11 boys at the reunion given that the total number
    of handshakes was 55 and each boy shook hands exactly once with each
    of the others. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 55 ∧ n = 11 := by
  sorry

#eval handshakes 11  -- This should output 55

end boys_at_reunion_l2573_257313


namespace green_hats_not_adjacent_probability_l2573_257343

def total_children : ℕ := 9
def green_hats : ℕ := 3

theorem green_hats_not_adjacent_probability :
  let total_arrangements := Nat.choose total_children green_hats
  let adjacent_arrangements := (total_children - green_hats + 1) + (total_children - 1) * (total_children - green_hats - 1)
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements = 5 / 14 := by
  sorry

end green_hats_not_adjacent_probability_l2573_257343


namespace segment_length_l2573_257398

/-- Given a line segment CD with points R and S on it, prove that CD has length 22.5 -/
theorem segment_length (C D R S : ℝ) : 
  R > C → -- R is to the right of C
  S > R → -- S is to the right of R
  D > S → -- D is to the right of S
  (R - C) / (D - R) = 1 / 4 → -- R divides CD in ratio 1:4
  (S - C) / (D - S) = 1 / 2 → -- S divides CD in ratio 1:2
  S - R = 3 → -- Length of RS is 3
  D - C = 22.5 := by -- Length of CD is 22.5
sorry


end segment_length_l2573_257398


namespace luna_bus_cost_l2573_257326

/-- The distance from city X to city Y in kilometers -/
def distance_XY : ℝ := 4500

/-- The cost per kilometer for bus travel in dollars -/
def bus_cost_per_km : ℝ := 0.20

/-- The total cost for Luna to bus from city X to city Y -/
def total_bus_cost : ℝ := distance_XY * bus_cost_per_km

/-- Theorem stating that the total bus cost for Luna to travel from X to Y is $900 -/
theorem luna_bus_cost : total_bus_cost = 900 := by
  sorry

end luna_bus_cost_l2573_257326


namespace min_sum_of_dimensions_l2573_257357

theorem min_sum_of_dimensions (l w h : ℕ) : 
  l > 0 → w > 0 → h > 0 → l * w * h = 2310 → 
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → a * b * c = 2310 → 
  l + w + h ≤ a + b + c → l + w + h = 42 := by sorry

end min_sum_of_dimensions_l2573_257357


namespace solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2573_257307

-- Define the equation
def equation (a : ℚ) (x : ℚ) : Prop :=
  |a| * x = |a + 1| - x

-- Theorem 1
theorem solve_for_a_when_x_is_zero :
  ∀ a : ℚ, equation a 0 → a = -1 :=
sorry

-- Theorem 2
theorem range_of_a_when_x_is_one :
  ∀ a : ℚ, equation a 1 → a ≥ 0 :=
sorry

end solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2573_257307


namespace rotate_point_D_l2573_257344

/-- Rotates a point (x, y) by 180 degrees around a center (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

theorem rotate_point_D :
  let d : ℝ × ℝ := (2, -3)
  let center : ℝ × ℝ := (3, -2)
  rotate180 d.1 d.2 center.1 center.2 = (4, -1) := by
sorry

end rotate_point_D_l2573_257344


namespace faye_score_l2573_257396

/-- Given a baseball team with the following properties:
  * The team has 5 players
  * The team scored a total of 68 points
  * 4 players scored 8 points each
  Prove that the remaining player (Faye) scored 36 points. -/
theorem faye_score (total_score : ℕ) (team_size : ℕ) (other_player_score : ℕ) :
  total_score = 68 →
  team_size = 5 →
  other_player_score = 8 →
  ∃ (faye_score : ℕ), faye_score = total_score - (team_size - 1) * other_player_score ∧
                      faye_score = 36 :=
by sorry

end faye_score_l2573_257396


namespace divisibility_equations_solutions_l2573_257308

theorem divisibility_equations_solutions :
  (∀ x : ℤ, (x - 1 ∣ x + 3) ↔ x ∈ ({-3, -1, 0, 2, 3, 5} : Set ℤ)) ∧
  (∀ x : ℤ, (x + 2 ∣ x^2 + 2) ↔ x ∈ ({-8, -5, -4, -3, -1, 0, 1, 4} : Set ℤ)) :=
by sorry

end divisibility_equations_solutions_l2573_257308


namespace inequality_proof_l2573_257372

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end inequality_proof_l2573_257372


namespace sin_cos_equality_solution_l2573_257390

theorem sin_cos_equality_solution :
  ∃ x : ℝ, x * (180 / π) = 9 ∧ Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x) := by
  sorry

end sin_cos_equality_solution_l2573_257390


namespace starting_lineup_theorem_l2573_257394

/-- The number of ways to choose a starting lineup from a basketball team. -/
def starting_lineup_choices (team_size : ℕ) (lineup_size : ℕ) (point_guard_count : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) (lineup_size - 1)

/-- Theorem: The number of ways to choose a starting lineup of 5 players
    from a team of 12, where one player must be the point guard and
    the other four positions are interchangeable, is equal to 3960. -/
theorem starting_lineup_theorem :
  starting_lineup_choices 12 5 1 = 3960 := by
  sorry

end starting_lineup_theorem_l2573_257394


namespace base7_246_to_base10_l2573_257363

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 7^2 + d1 * 7^1 + d0 * 7^0

/-- The base 10 representation of 246 in base 7 is 132 -/
theorem base7_246_to_base10 : base7ToBase10 2 4 6 = 132 := by
  sorry

end base7_246_to_base10_l2573_257363


namespace boxes_ordered_correct_l2573_257336

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 300

/-- Represents the fraction of stock sold -/
def fraction_sold : ℚ := 3/4

/-- Represents the number of unsold apples -/
def unsold_apples : ℕ := 750

/-- Calculates the number of boxes ordered each week -/
def boxes_ordered : ℕ := 10

/-- Proves that the number of boxes ordered is correct given the conditions -/
theorem boxes_ordered_correct :
  (1 - fraction_sold) * (apples_per_box * boxes_ordered) = unsold_apples := by sorry

end boxes_ordered_correct_l2573_257336


namespace cube_volume_problem_l2573_257321

theorem cube_volume_problem (cube_a_volume : ℝ) (surface_area_ratio : ℝ) :
  cube_a_volume = 8 →
  surface_area_ratio = 3 →
  ∃ (cube_b_volume : ℝ),
    (6 * (cube_a_volume ^ (1/3))^2) * surface_area_ratio = 6 * (cube_b_volume ^ (1/3))^2 ∧
    cube_b_volume = 24 * Real.sqrt 3 :=
by sorry

end cube_volume_problem_l2573_257321


namespace male_students_count_l2573_257324

theorem male_students_count (total : ℕ) (male : ℕ) (female : ℕ) :
  total = 48 →
  female = (4 * male) / 5 + 3 →
  total = male + female →
  male = 25 := by
sorry

end male_students_count_l2573_257324


namespace part1_correct_part2_correct_l2573_257340

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (-3*m - 4, 2 + m)

-- Define point Q
def Q : ℝ × ℝ := (5, 8)

-- Theorem for part 1
theorem part1_correct :
  ∃ m : ℝ, P m = (-10, 4) ∧ (P m).2 = 4 := by sorry

-- Theorem for part 2
theorem part2_correct :
  ∃ m : ℝ, P m = (5, -1) ∧ (P m).1 = Q.1 := by sorry

end part1_correct_part2_correct_l2573_257340


namespace root_in_interval_implies_a_range_l2573_257306

theorem root_in_interval_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : a < 0) 
  (h3 : ∃ x ∈ Set.Ioo 0 1, f x = 0) : 
  -2 < a ∧ a < 0 := by
sorry

end root_in_interval_implies_a_range_l2573_257306


namespace arithmetic_comparisons_l2573_257345

theorem arithmetic_comparisons : 
  (25 + 45 = 45 + 25) ∧ 
  (56 - 28 < 65 - 28) ∧ 
  (22 * 41 = 41 * 22) ∧ 
  (50 - 32 > 50 - 23) := by
sorry

end arithmetic_comparisons_l2573_257345


namespace electricity_billing_theorem_l2573_257316

/-- Represents a three-tariff meter reading --/
structure MeterReading where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_ordered : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f

/-- Represents tariff prices --/
structure TariffPrices where
  t₁ : ℝ
  t₂ : ℝ
  t₃ : ℝ

/-- Calculates the maximum additional payment --/
def maxAdditionalPayment (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Calculates the expected value of the difference --/
def expectedDifference (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Main theorem --/
theorem electricity_billing_theorem (reading : MeterReading) (prices : TariffPrices) :
  let actualPayment := 660.72
  prices.t₁ = 4.03 ∧ prices.t₂ = 1.01 ∧ prices.t₃ = 3.39 →
  reading.a = 1214 ∧ reading.b = 1270 ∧ reading.c = 1298 ∧
  reading.d = 1337 ∧ reading.e = 1347 ∧ reading.f = 1402 →
  maxAdditionalPayment reading prices actualPayment = 397.34 ∧
  expectedDifference reading prices actualPayment = 19.30 :=
sorry

end electricity_billing_theorem_l2573_257316


namespace least_five_digit_prime_congruent_to_7_mod_20_l2573_257371

theorem least_five_digit_prime_congruent_to_7_mod_20 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (n % 20 = 7) ∧              -- congruent to 7 (mod 20)
  Nat.Prime n ∧               -- prime number
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → (m % 20 = 7) → Nat.Prime m → m ≥ n) ∧
  n = 10127 := by
sorry

end least_five_digit_prime_congruent_to_7_mod_20_l2573_257371


namespace hyperbola_condition_l2573_257333

-- Define the equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (3 - k) + y^2 / (k - 2) = 1

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation x y k ∧ (3 - k) * (k - 2) < 0

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k ↔ k < 2 ∨ k > 3 :=
by sorry

end hyperbola_condition_l2573_257333


namespace cryptarithm_solution_l2573_257383

-- Define the cryptarithm equation
def cryptarithm (A B C : ℕ) : Prop :=
  A < 10 ∧ B < 10 ∧ C < 10 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  100 * C + 10 * B + A + 100 * A + 10 * A + A = 10 * B + A

-- Theorem statement
theorem cryptarithm_solution :
  ∃! (A B C : ℕ), cryptarithm A B C ∧ A = 5 ∧ B = 9 ∧ C = 3 := by
  sorry

end cryptarithm_solution_l2573_257383


namespace joan_marbles_l2573_257378

theorem joan_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : total_marbles = 12) :
  total_marbles - mary_marbles = 3 :=
by sorry

end joan_marbles_l2573_257378


namespace last_two_digits_sum_factorials_15_l2573_257304

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by
  sorry

end last_two_digits_sum_factorials_15_l2573_257304


namespace parallel_lines_perpendicular_lines_l2573_257347

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k) (y + k * (a / 2))

-- Define perpendicular lines
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l1 a x₁ y₁ ∧ l2 a x₂ y₂ → (x₂ - x₁) * (a * (x₂ - x₁) + 2 * (y₂ - y₁)) + (y₂ - y₁) * ((a - 1) * (x₂ - x₁) + (y₂ - y₁)) = 0

-- Theorem for parallel lines
theorem parallel_lines : ∀ a : ℝ, parallel a → a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ a : ℝ, perpendicular a → a = 2/3 :=
sorry

end parallel_lines_perpendicular_lines_l2573_257347


namespace determinant_in_terms_of_r_s_t_l2573_257382

theorem determinant_in_terms_of_r_s_t (r s t : ℝ) (a b c : ℝ) : 
  (a^3 - r*a^2 + s*a - t = 0) →
  (b^3 - r*b^2 + s*b - t = 0) →
  (c^3 - r*c^2 + s*c - t = 0) →
  (a + b + c = r) →
  (a*b + a*c + b*c = s) →
  (a*b*c = t) →
  Matrix.det !![2+a, 2, 2; 2, 2+b, 2; 2, 2, 2+c] = t - 2*s := by
sorry

end determinant_in_terms_of_r_s_t_l2573_257382


namespace price_reduction_equation_l2573_257369

/-- Represents the price reduction percentage -/
def x : ℝ := sorry

/-- The original price of the medicine -/
def original_price : ℝ := 25

/-- The final price of the medicine after two reductions -/
def final_price : ℝ := 16

/-- Theorem stating the relationship between the original price, 
    final price, and the reduction percentage -/
theorem price_reduction_equation : 
  original_price * (1 - x)^2 = final_price := by sorry

end price_reduction_equation_l2573_257369


namespace product_of_four_consecutive_integers_divisible_by_24_l2573_257331

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k :=
by sorry

end product_of_four_consecutive_integers_divisible_by_24_l2573_257331


namespace sum_of_tenth_powers_l2573_257359

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem sum_of_tenth_powers (a b : ℝ) 
  (sum1 : a + b = 1)
  (sum2 : a^2 + b^2 = 3)
  (sum3 : a^3 + b^3 = 4)
  (sum4 : a^4 + b^4 = 7)
  (sum5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end sum_of_tenth_powers_l2573_257359


namespace rectangular_plot_dimensions_l2573_257377

theorem rectangular_plot_dimensions (area : ℝ) (fence_length : ℝ) :
  area = 800 ∧ fence_length = 100 →
  ∃ (length width : ℝ),
    (length * width = area ∧
     2 * length + width = fence_length) ∧
    ((length = 40 ∧ width = 20) ∨ (length = 10 ∧ width = 80)) := by
  sorry

end rectangular_plot_dimensions_l2573_257377


namespace missing_number_is_36_l2573_257302

def known_numbers : List ℕ := [1, 22, 23, 24, 25, 27, 2]

theorem missing_number_is_36 (mean : ℚ) (total_count : ℕ) (h_mean : mean = 20) (h_count : total_count = 8) :
  ∃ x : ℕ, (x :: known_numbers).sum / total_count = mean :=
sorry

end missing_number_is_36_l2573_257302


namespace solve_probability_problem_l2573_257332

def probability_problem (p_man : ℚ) (p_wife : ℚ) : Prop :=
  p_man = 1/4 ∧ p_wife = 1/3 →
  (1 - p_man) * (1 - p_wife) = 1/2

theorem solve_probability_problem : probability_problem (1/4) (1/3) := by
  sorry

end solve_probability_problem_l2573_257332


namespace smallest_n_congruence_l2573_257317

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k < n, (7^k : ℤ) % 4 ≠ k^7 % 4) ∧ (7^n : ℤ) % 4 = n^7 % 4 ↔ n = 3 :=
sorry

end smallest_n_congruence_l2573_257317


namespace multiplication_fraction_equality_l2573_257355

theorem multiplication_fraction_equality : 7 * (1 / 21) * 42 = 14 := by
  sorry

end multiplication_fraction_equality_l2573_257355


namespace age_sum_problem_l2573_257387

theorem age_sum_problem (a b c : ℕ+) : 
  b = c →                   -- The twins have the same age
  a < b →                   -- Kiana is younger than her brothers
  a * b * c = 256 →         -- The product of their ages is 256
  a + b + c = 20 :=         -- The sum of their ages is 20
by sorry

end age_sum_problem_l2573_257387


namespace prime_remainder_mod_30_l2573_257380

theorem prime_remainder_mod_30 (p : ℕ) (hp : Prime p) : 
  ∃ (r : ℕ), p % 30 = r ∧ (r = 1 ∨ (Prime r ∧ r < 30)) := by
  sorry

end prime_remainder_mod_30_l2573_257380


namespace gold_silver_alloy_composition_l2573_257339

/-- Prove the composition of a gold-silver alloy given its properties -/
theorem gold_silver_alloy_composition
  (total_mass : ℝ)
  (total_volume : ℝ)
  (density_gold : ℝ)
  (density_silver : ℝ)
  (h_total_mass : total_mass = 13.85)
  (h_total_volume : total_volume = 0.9)
  (h_density_gold : density_gold = 19.3)
  (h_density_silver : density_silver = 10.5) :
  ∃ (mass_gold mass_silver : ℝ),
    mass_gold + mass_silver = total_mass ∧
    mass_gold / density_gold + mass_silver / density_silver = total_volume ∧
    mass_gold = 9.65 ∧
    mass_silver = 4.2 := by
  sorry

end gold_silver_alloy_composition_l2573_257339


namespace m_range_l2573_257320

def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- ¬p is a sufficient but not necessary condition for ¬q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ (∃ x, ¬(q x m) ∧ p x)

theorem m_range :
  ∀ m, sufficient_not_necessary m ↔ 2 ≤ m ∧ m ≤ 4 :=
sorry

end m_range_l2573_257320


namespace used_car_selections_l2573_257314

theorem used_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 12)
  (h2 : num_clients = 9)
  (h3 : selections_per_client = 4) :
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end used_car_selections_l2573_257314


namespace hyperbola_sum_l2573_257379

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 ∧ 
  k = 0 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by
sorry

end hyperbola_sum_l2573_257379


namespace expression_simplification_l2573_257318

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end expression_simplification_l2573_257318


namespace largest_of_three_l2573_257303

theorem largest_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p*q + p*r + q*r = -6)
  (prod_eq : p*q*r = -18) :
  max p (max q r) = Real.sqrt 6 := by
sorry

end largest_of_three_l2573_257303


namespace investment_income_is_660_l2573_257366

/-- Calculates the total annual income from an investment split between a savings account and bonds. -/
def totalAnnualIncome (totalInvestment : ℝ) (savingsAmount : ℝ) (savingsRate : ℝ) (bondRate : ℝ) : ℝ :=
  let bondAmount := totalInvestment - savingsAmount
  savingsAmount * savingsRate + bondAmount * bondRate

/-- Proves that the total annual income from the given investment scenario is $660. -/
theorem investment_income_is_660 :
  totalAnnualIncome 10000 6000 0.05 0.09 = 660 := by
  sorry

#eval totalAnnualIncome 10000 6000 0.05 0.09

end investment_income_is_660_l2573_257366


namespace negation_of_universal_proposition_l2573_257346

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 4) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 4) := by
  sorry

end negation_of_universal_proposition_l2573_257346
