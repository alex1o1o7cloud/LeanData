import Mathlib

namespace two_digit_number_problem_l3888_388867

theorem two_digit_number_problem (M N : ℕ) (h1 : M < 10) (h2 : N < 10) (h3 : N > M) :
  let x := 10 * N + M
  let y := 10 * M + N
  (x + y = 11 * (x - y)) → (M = 4 ∧ N = 5) := by
sorry


end two_digit_number_problem_l3888_388867


namespace total_red_balloons_l3888_388810

theorem total_red_balloons (sam_initial : ℝ) (fred_received : ℝ) (dan_balloons : ℝ)
  (h1 : sam_initial = 46.0)
  (h2 : fred_received = 10.0)
  (h3 : dan_balloons = 16.0) :
  sam_initial - fred_received + dan_balloons = 52.0 := by
sorry

end total_red_balloons_l3888_388810


namespace journey_time_ratio_l3888_388806

/-- Proves that for a journey of 288 km, if the original time taken is 6 hours
    and the new speed is 32 kmph, then the ratio of the new time to the original time is 3:2. -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry


end journey_time_ratio_l3888_388806


namespace radical_simplification_l3888_388803

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (20 * q^3) * Real.sqrt (12 * q^5) = 60 * q^4 * Real.sqrt q :=
by sorry

end radical_simplification_l3888_388803


namespace decimal_119_equals_base6_315_l3888_388841

/-- Converts a natural number to its base 6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of base 6 digits to its decimal (base 10) value -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => 6 * acc + d) 0

theorem decimal_119_equals_base6_315 : toBase6 119 = [3, 1, 5] ∧ fromBase6 [3, 1, 5] = 119 := by
  sorry

#eval toBase6 119  -- Should output [3, 1, 5]
#eval fromBase6 [3, 1, 5]  -- Should output 119

end decimal_119_equals_base6_315_l3888_388841


namespace intersection_and_angle_condition_l3888_388833

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y + 1 = 0
def l2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0
def l3 (x y : ℝ) : Prop := 3 * x + 2 * y + 1 = 0

-- Define the result lines
def result1 (x y : ℝ) : Prop := x + 5 * y + 5 = 0
def result2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0

-- Define the 45° angle condition
def angle_45_deg (m1 m2 : ℝ) : Prop := (m1 - m2) / (1 + m1 * m2) = 1 || (m1 - m2) / (1 + m1 * m2) = -1

-- Main theorem
theorem intersection_and_angle_condition :
  ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧
  (∃ (m : ℝ), (angle_45_deg m (-3/2)) ∧
    ((result1 x y ∧ m = -1/5) ∨ (result2 x y ∧ m = 5))) :=
sorry

end intersection_and_angle_condition_l3888_388833


namespace vegetable_count_l3888_388819

/-- The total number of vegetables in the supermarket -/
def total_vegetables (cucumbers carrots tomatoes radishes : ℕ) : ℕ :=
  cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables given the conditions -/
theorem vegetable_count :
  ∀ (cucumbers carrots tomatoes radishes : ℕ),
    cucumbers = 58 →
    cucumbers = carrots + 24 →
    cucumbers = tomatoes - 49 →
    radishes = carrots →
    total_vegetables cucumbers carrots tomatoes radishes = 233 := by
  sorry

end vegetable_count_l3888_388819


namespace line_parameterization_l3888_388897

/-- Given a line y = (2/3)x - 5 parameterized by (x, y) = (-6, p) + t(m, 7),
    prove that p = -9 and m = 21/2 -/
theorem line_parameterization (x y t : ℝ) (p m : ℝ) :
  (y = (2/3) * x - 5) →
  (∃ t, x = -6 + t * m ∧ y = p + t * 7) →
  (p = -9 ∧ m = 21/2) := by
  sorry

end line_parameterization_l3888_388897


namespace discount_rate_calculation_l3888_388827

theorem discount_rate_calculation (marked_price selling_price : ℝ) 
  (h1 : marked_price = 80)
  (h2 : selling_price = 68) :
  (marked_price - selling_price) / marked_price * 100 = 15 := by
sorry

end discount_rate_calculation_l3888_388827


namespace min_sum_a_b_l3888_388892

theorem min_sum_a_b (a b : ℕ+) (h : 45 * a + b = 2021) : 
  (∀ (x y : ℕ+), 45 * x + y = 2021 → a + b ≤ x + y) ∧ a + b = 85 := by
  sorry

end min_sum_a_b_l3888_388892


namespace cannot_determine_jake_peaches_l3888_388874

def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

structure Jake where
  peaches : ℕ
  apples : ℕ
  fewer_peaches : peaches < steven_peaches
  more_apples : apples = steven_apples + 3

theorem cannot_determine_jake_peaches : ∀ (jake : Jake), ∃ (jake' : Jake), jake.peaches ≠ jake'.peaches := by
  sorry

end cannot_determine_jake_peaches_l3888_388874


namespace koala_fiber_consumption_l3888_388829

/-- The absorption rate of fiber for koalas -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day, 
    then the total amount of fiber it ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end koala_fiber_consumption_l3888_388829


namespace equation_solution_l3888_388854

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end equation_solution_l3888_388854


namespace hash_2_neg1_4_l3888_388853

def hash (a b c : ℝ) : ℝ := a * b^2 - 3 * a - 5 * c

theorem hash_2_neg1_4 : hash 2 (-1) 4 = -24 := by
  sorry

end hash_2_neg1_4_l3888_388853


namespace ellipse_canonical_equation_l3888_388855

/-- Proves that an ellipse with given minor axis length and distance between foci has the specified canonical equation -/
theorem ellipse_canonical_equation 
  (minor_axis : ℝ) 
  (foci_distance : ℝ) 
  (h_minor : minor_axis = 6) 
  (h_foci : foci_distance = 8) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 25 + y^2 / 9 = 1)) :=
sorry

end ellipse_canonical_equation_l3888_388855


namespace triangle_side_ratio_l3888_388811

theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c) :
  2 * b^2 = a^2 + c^2 := by sorry

end triangle_side_ratio_l3888_388811


namespace female_officers_count_l3888_388831

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) (female_percent_of_total : ℚ) :
  total_on_duty = 150 →
  female_on_duty_percent = 25 / 100 →
  female_percent_of_total = 40 / 100 →
  ∃ (total_female : ℕ), total_female = 240 ∧ 
    (female_on_duty_percent * total_female : ℚ) = (female_percent_of_total * total_on_duty : ℚ) := by
  sorry

end female_officers_count_l3888_388831


namespace a_5_equals_31_l3888_388893

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_equals_31 (a : ℕ → ℝ) :
  geometric_sequence (λ n => 1 + a n) →
  (∀ n : ℕ, (1 + a (n + 1)) = 2 * (1 + a n)) →
  a 1 = 1 →
  a 5 = 31 := by
sorry

end a_5_equals_31_l3888_388893


namespace arithmetic_sequence_common_difference_range_l3888_388877

/-- For a positive arithmetic sequence with a_3 = 2, the common difference d is in the range [0, 1). -/
theorem arithmetic_sequence_common_difference_range 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 2) :
  ∃ d, (∀ n, a (n + 1) = a n + d) ∧ 0 ≤ d ∧ d < 1 := by
  sorry

end arithmetic_sequence_common_difference_range_l3888_388877


namespace income_ratio_problem_l3888_388864

/-- Given two persons P1 and P2 with incomes and expenditures, prove their income ratio --/
theorem income_ratio_problem (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℚ) : 
  income_P1 = 3000 →
  expenditure_P1 / expenditure_P2 = 3 / 2 →
  income_P1 - expenditure_P1 = 1200 →
  income_P2 - expenditure_P2 = 1200 →
  income_P1 / income_P2 = 5 / 4 := by
  sorry

end income_ratio_problem_l3888_388864


namespace smallest_dual_base_palindrome_l3888_388861

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => acc * base + d) 0 ∧ digits = digits.reverse

/-- The theorem stating that 105 is the smallest natural number greater than 20 
    that is a palindrome in both base 14 and base 20 -/
theorem smallest_dual_base_palindrome :
  ∀ (N : ℕ), N > 20 → isPalindromeInBase N 14 → isPalindromeInBase N 20 → N ≥ 105 :=
by
  sorry

#check smallest_dual_base_palindrome

end smallest_dual_base_palindrome_l3888_388861


namespace min_distance_between_graphs_l3888_388814

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (-2*x + 1)

/-- The logarithmic function -/
noncomputable def g (x : ℝ) : ℝ := (Real.log (-x - 1) - 3) / 2

/-- The symmetry line -/
noncomputable def l (x : ℝ) : ℝ := -x - 1

/-- Theorem stating the minimum distance between points on the two graphs -/
theorem min_distance_between_graphs :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = f P.1) ∧ 
    (Q.2 = g Q.1) ∧
    (∀ (P' Q' : ℝ × ℝ), 
      P'.2 = f P'.1 → 
      Q'.2 = g Q'.1 → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 2 * (4 + Real.log 2)) / 2 :=
by sorry

end min_distance_between_graphs_l3888_388814


namespace consecutive_color_draw_probability_l3888_388875

def num_tan_chips : ℕ := 3
def num_pink_chips : ℕ := 2
def num_violet_chips : ℕ := 4
def total_chips : ℕ := num_tan_chips + num_pink_chips + num_violet_chips

theorem consecutive_color_draw_probability :
  (Nat.factorial num_tan_chips * Nat.factorial num_pink_chips * 
   Nat.factorial num_violet_chips * Nat.factorial 3) / 
  Nat.factorial total_chips = 1 / 210 := by
  sorry

end consecutive_color_draw_probability_l3888_388875


namespace injective_function_inequality_l3888_388886

theorem injective_function_inequality (f : Set.Icc 0 1 → ℝ) 
  (h_inj : Function.Injective f) (h_sum : f 0 + f 1 = 1) :
  ∃ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ ∧ 2 * f x₁ < f x₂ + 1/2 := by
  sorry

end injective_function_inequality_l3888_388886


namespace simplify_radical_expression_l3888_388889

theorem simplify_radical_expression (y : ℝ) (h : y > 0) :
  (32 * y) ^ (1/4) * (50 * y) ^ (1/4) + (18 * y) ^ (1/4) = 
  10 * (8 * y^2) ^ (1/4) + 3 * (2 * y) ^ (1/4) :=
by sorry

end simplify_radical_expression_l3888_388889


namespace circle_equation_l3888_388816

-- Define the circle C
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def TangentLine := {(x, y) : ℝ × ℝ | 3*x + 4*y + 4 = 0}

-- Theorem statement
theorem circle_equation (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ (p : ℝ × ℝ), p ∈ Circle a ∧ p ∈ TangentLine) : 
  Circle a = Circle 2 := by
sorry

end circle_equation_l3888_388816


namespace bob_tv_width_is_90_l3888_388838

/-- The width of Bob's TV -/
def bob_tv_width : ℝ := 90

/-- The height of Bob's TV -/
def bob_tv_height : ℝ := 60

/-- The width of Bill's TV -/
def bill_tv_width : ℝ := 100

/-- The height of Bill's TV -/
def bill_tv_height : ℝ := 48

/-- Weight of TV per square inch in ounces -/
def tv_weight_per_sq_inch : ℝ := 4

/-- Ounces per pound -/
def oz_per_pound : ℝ := 16

/-- Weight difference between Bob's and Bill's TVs in pounds -/
def weight_difference : ℝ := 150

theorem bob_tv_width_is_90 :
  bob_tv_width = 90 :=
by
  sorry

#check bob_tv_width_is_90

end bob_tv_width_is_90_l3888_388838


namespace tens_digit_of_9_to_1024_l3888_388888

theorem tens_digit_of_9_to_1024 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^1024 ≡ n [ZMOD 100] ∧ (n / 10) % 10 = 6 := by
  sorry

end tens_digit_of_9_to_1024_l3888_388888


namespace two_natural_numbers_problem_l3888_388842

theorem two_natural_numbers_problem :
  ∃ (x y : ℕ), x > y ∧ 
    x + y = 5 * (x - y) ∧
    x * y = 24 * (x - y) ∧
    x = 12 ∧ y = 8 := by
  sorry

end two_natural_numbers_problem_l3888_388842


namespace expression_evaluation_l3888_388852

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11/36 := by
  sorry

end expression_evaluation_l3888_388852


namespace arccos_value_from_arcsin_inequality_l3888_388843

theorem arccos_value_from_arcsin_inequality (a b : ℝ) :
  Real.arcsin (1 + a^2) - Real.arcsin ((b - 1)^2) ≥ π / 2 →
  Real.arccos (a^2 - b^2) = π := by
  sorry

end arccos_value_from_arcsin_inequality_l3888_388843


namespace school_outing_problem_l3888_388832

theorem school_outing_problem (x : ℕ) : 
  (3 * x + 16 = 5 * (x - 1) + 1) → (3 * x + 16 = 46) := by
  sorry

end school_outing_problem_l3888_388832


namespace soccer_team_average_goals_l3888_388851

/-- Calculates the average number of goals per game for a soccer team -/
def average_goals_per_game (slices_per_pizza : ℕ) (pizzas_bought : ℕ) (games_played : ℕ) : ℚ :=
  (slices_per_pizza * pizzas_bought : ℚ) / games_played

/-- Theorem: Given the conditions, the average number of goals per game is 9 -/
theorem soccer_team_average_goals :
  let slices_per_pizza : ℕ := 12
  let pizzas_bought : ℕ := 6
  let games_played : ℕ := 8
  average_goals_per_game slices_per_pizza pizzas_bought games_played = 9 := by
sorry

#eval average_goals_per_game 12 6 8

end soccer_team_average_goals_l3888_388851


namespace eight_beads_two_identical_arrangements_l3888_388881

/-- The number of unique arrangements of n distinct beads, including k identical beads, on a bracelet, considering rotational and reflectional symmetry -/
def uniqueBraceletArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / k.factorial) / (2 * n)

theorem eight_beads_two_identical_arrangements :
  uniqueBraceletArrangements 8 2 = 1260 := by
  sorry

end eight_beads_two_identical_arrangements_l3888_388881


namespace largest_valid_number_l3888_388825

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- all digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- digits in ascending order

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7089 :=
by sorry

end largest_valid_number_l3888_388825


namespace smallest_s_for_array_l3888_388826

theorem smallest_s_for_array (m n : ℕ+) : ∃ (s : ℕ+),
  (∀ (s' : ℕ+), s' < s → ¬∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s')) ∧
  (∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s)) ∧
  s = m + n - Nat.gcd m n :=
by sorry

end smallest_s_for_array_l3888_388826


namespace product_of_monomials_l3888_388830

theorem product_of_monomials (x y : ℝ) : 2 * x * (-3 * x^2 * y^3) = -6 * x^3 * y^3 := by
  sorry

end product_of_monomials_l3888_388830


namespace area_ratio_is_five_sevenths_l3888_388885

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def AD : ℝ := 13
def AO : ℝ := 13
def OB : ℝ := 13
def BC : ℝ := 13
def AB : ℝ := 15
def DO : ℝ := 15
def OC : ℝ := 15

-- Define the conditions
axiom triangle_dao_isosceles : AO = AD
axiom triangle_aob_isosceles : AO = OB
axiom triangle_obc_isosceles : OB = BC
axiom p_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom op_perpendicular_ab : (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0
axiom x_midpoint_ad : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom y_midpoint_bc : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- State the theorem
theorem area_ratio_is_five_sevenths :
  area_ABYX / area_XYCD = 5 / 7 := by sorry

end area_ratio_is_five_sevenths_l3888_388885


namespace min_c_value_l3888_388817

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2021 ∧ p.2 = |p.1 - a| + |p.1 - b| + |p.1 - c|) :
  c ≥ 1011 :=
by sorry

end min_c_value_l3888_388817


namespace cube_root_five_sixteenths_l3888_388879

theorem cube_root_five_sixteenths :
  (5 / 16 : ℝ)^(1/3) = (5 : ℝ)^(1/3) / 2^(4/3) := by sorry

end cube_root_five_sixteenths_l3888_388879


namespace smallest_twin_egg_number_l3888_388880

def is_twin_egg_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ n = 1000 * a + 100 * b + 10 * b + a

def F (m : ℕ) : ℚ :=
  let m' := (m % 100) * 100 + (m / 100)
  (m - m') / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    ∃ (k : ℕ), F m / 54 = k^2 ∧
    ∀ (n : ℕ), is_twin_egg_number n → (∃ (l : ℕ), F n / 54 = l^2) → m ≤ n ∧
    m = 7117 :=
sorry

end smallest_twin_egg_number_l3888_388880


namespace completing_square_sum_l3888_388869

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 + 6*x - 1 = 0 ↔ (x + a)^2 = b) → a + b = 13 := by
  sorry

end completing_square_sum_l3888_388869


namespace claire_orange_price_l3888_388868

-- Define the given quantities
def liam_oranges : ℕ := 40
def liam_price : ℚ := 2.5 / 2
def claire_oranges : ℕ := 30
def total_savings : ℚ := 86

-- Define Claire's price per orange
def claire_price : ℚ := (total_savings - (liam_oranges : ℚ) * liam_price) / (claire_oranges : ℚ)

-- Theorem statement
theorem claire_orange_price : claire_price = 1.2 := by
  sorry

end claire_orange_price_l3888_388868


namespace rectangular_field_shortcut_l3888_388844

theorem rectangular_field_shortcut (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) :
  x + y - Real.sqrt (x^2 + y^2) = x →
  y / x = 1 / 2 := by
sorry

end rectangular_field_shortcut_l3888_388844


namespace product_sum_relation_l3888_388845

theorem product_sum_relation (a b N : ℤ) : 
  b = 7 → 
  b - a = 4 → 
  a * b = 2 * (a + b) + N → 
  N = 1 := by
sorry

end product_sum_relation_l3888_388845


namespace functional_equation_solution_l3888_388873

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by sorry

end functional_equation_solution_l3888_388873


namespace problem_statement_l3888_388899

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2014 = 1 := by
  sorry

end problem_statement_l3888_388899


namespace arithmetic_equation_l3888_388884

theorem arithmetic_equation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_equation_l3888_388884


namespace binomial_15_13_l3888_388822

theorem binomial_15_13 : Nat.choose 15 13 = 105 := by
  sorry

end binomial_15_13_l3888_388822


namespace election_votes_proof_l3888_388836

theorem election_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (winner_percentage : ℚ) :
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  winner_percentage = 11/20 →
  let valid_votes := total_votes - (invalid_percentage * total_votes).num
  let winner_votes := (winner_percentage * valid_votes).num
  valid_votes - winner_votes = 2520 :=
by sorry

end election_votes_proof_l3888_388836


namespace total_savings_ten_sets_l3888_388848

-- Define the cost of 2 packs
def cost_two_packs : ℚ := 2.5

-- Define the cost of an individual pack
def cost_individual : ℚ := 1.3

-- Define the number of sets
def num_sets : ℕ := 10

-- Theorem statement
theorem total_savings_ten_sets : 
  let cost_per_pack := cost_two_packs / 2
  let savings_per_pack := cost_individual - cost_per_pack
  let total_packs := num_sets * 2
  savings_per_pack * total_packs = 1 := by sorry

end total_savings_ten_sets_l3888_388848


namespace existence_of_counterexample_l3888_388870

theorem existence_of_counterexample : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end existence_of_counterexample_l3888_388870


namespace faculty_reduction_l3888_388895

theorem faculty_reduction (initial_faculty : ℝ) (reduction_percentage : ℝ) : 
  initial_faculty = 243.75 →
  reduction_percentage = 20 →
  initial_faculty * (1 - reduction_percentage / 100) = 195 := by
  sorry

end faculty_reduction_l3888_388895


namespace square_sum_equality_l3888_388860

theorem square_sum_equality : 784 + 2 * 14 * 7 + 49 = 1225 := by
  sorry

end square_sum_equality_l3888_388860


namespace noemi_initial_amount_l3888_388804

def initial_amount (roulette_loss blackjack_loss remaining : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + remaining

theorem noemi_initial_amount :
  initial_amount 400 500 800 = 1700 := by
  sorry

end noemi_initial_amount_l3888_388804


namespace negation_of_implication_l3888_388834

theorem negation_of_implication (x y : ℝ) :
  ¬(xy = 0 → x = 0 ∨ y = 0) ↔ (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) := by
  sorry

end negation_of_implication_l3888_388834


namespace sin_n_eq_cos_810_l3888_388856

theorem sin_n_eq_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (810 * π / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end sin_n_eq_cos_810_l3888_388856


namespace circles_intersection_parallel_lines_l3888_388837

-- Define the basic geometric objects
variable (Circle1 Circle2 : Set (ℝ × ℝ))
variable (M K A B C D : ℝ × ℝ)

-- Define the conditions
axiom intersect_points : M ∈ Circle1 ∧ M ∈ Circle2 ∧ K ∈ Circle1 ∧ K ∈ Circle2
axiom line_AB : M.1 * B.2 - M.2 * B.1 = A.1 * B.2 - A.2 * B.1
axiom line_CD : K.1 * D.2 - K.2 * D.1 = C.1 * D.2 - C.2 * D.1
axiom A_in_Circle1 : A ∈ Circle1
axiom B_in_Circle2 : B ∈ Circle2
axiom C_in_Circle1 : C ∈ Circle1
axiom D_in_Circle2 : D ∈ Circle2

-- Define parallel lines
def parallel (p q r s : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.2 - s.2) = (p.2 - q.2) * (r.1 - s.1)

-- State the theorem
theorem circles_intersection_parallel_lines :
  parallel A C B D :=
sorry

end circles_intersection_parallel_lines_l3888_388837


namespace trigonometric_simplification_l3888_388835

theorem trigonometric_simplification (x : ℝ) : 
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos (2 * π / 3 - x) = 0 := by
  sorry

end trigonometric_simplification_l3888_388835


namespace midpoint_chain_l3888_388812

/-- Given a line segment AB with multiple midpoints, prove that AB = 64 when AG = 2 -/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 2) →        -- AG = 2
  (B - A = 64) :=      -- AB = 64
by sorry

end midpoint_chain_l3888_388812


namespace savings_difference_l3888_388859

/-- Represents the price of a book in dollars -/
def book_price : ℝ := 25

/-- Represents the discount percentage for Discount A -/
def discount_a_percentage : ℝ := 0.4

/-- Represents the fixed discount amount for Discount B in dollars -/
def discount_b_amount : ℝ := 5

/-- Calculates the total cost with Discount A -/
def total_cost_a : ℝ := book_price + (book_price * (1 - discount_a_percentage))

/-- Calculates the total cost with Discount B -/
def total_cost_b : ℝ := book_price + (book_price - discount_b_amount)

/-- Theorem stating the difference in savings between Discount A and Discount B -/
theorem savings_difference : total_cost_b - total_cost_a = 5 := by
  sorry

end savings_difference_l3888_388859


namespace gym_equipment_cost_l3888_388821

/-- Calculates the total cost in dollars including sales tax for gym equipment purchase -/
def total_cost_with_tax (squat_rack_cost : ℝ) (barbell_fraction : ℝ) (weights_cost : ℝ) 
  (exchange_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let barbell_cost := squat_rack_cost * barbell_fraction
  let total_euro := squat_rack_cost + barbell_cost + weights_cost
  let total_dollar := total_euro * exchange_rate
  let tax := total_dollar * tax_rate
  total_dollar + tax

/-- Theorem stating the total cost of gym equipment including tax -/
theorem gym_equipment_cost : 
  total_cost_with_tax 2500 0.1 750 1.15 0.06 = 4266.50 := by
  sorry

end gym_equipment_cost_l3888_388821


namespace counterexample_exists_l3888_388862

theorem counterexample_exists : ∃ (a b : ℝ), (a + b < 0) ∧ ¬(a < 0 ∧ b < 0) := by
  sorry

end counterexample_exists_l3888_388862


namespace number_of_boys_l3888_388878

def total_students : ℕ := 1150

def is_valid_distribution (boys : ℕ) : Prop :=
  let girls := (boys * 100) / total_students
  boys + girls = total_students

theorem number_of_boys : ∃ (boys : ℕ), boys = 1058 ∧ is_valid_distribution boys := by
  sorry

end number_of_boys_l3888_388878


namespace birdhouse_distance_l3888_388801

/-- The distance flown by objects in a tornado scenario -/
def tornado_scenario (car_distance : ℕ) : Prop :=
  let lawn_chair_distance := 2 * car_distance
  let birdhouse_distance := 3 * lawn_chair_distance
  car_distance = 200 ∧ birdhouse_distance = 1200

/-- Theorem stating that in the given scenario, the birdhouse flew 1200 feet -/
theorem birdhouse_distance : tornado_scenario 200 := by
  sorry

end birdhouse_distance_l3888_388801


namespace bubble_sort_correct_l3888_388815

def bubbleSort (xs : List Int) : List Int :=
  let rec pass : List Int → List Int
    | [] => []
    | [x] => [x]
    | x :: y :: rest => if x <= y then x :: pass (y :: rest) else y :: pass (x :: rest)
  let rec sort (xs : List Int) (n : Nat) : List Int :=
    if n = 0 then xs else sort (pass xs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubbleSort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end bubble_sort_correct_l3888_388815


namespace basketball_handshakes_l3888_388809

/-- The number of handshakes in a basketball game -/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let inter_team := team_size * team_size * (num_teams - 1) / 2
  let intra_team := num_teams * (team_size * (team_size - 1) / 2)
  let with_referees := num_teams * team_size * num_referees
  inter_team + intra_team + with_referees

/-- Theorem stating the total number of handshakes in the specific basketball game scenario -/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 102 := by
  sorry

end basketball_handshakes_l3888_388809


namespace no_common_roots_l3888_388850

theorem no_common_roots (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^4 + b*x + c = 0) ∧ (x^4 + a*x + d = 0) :=
by sorry

end no_common_roots_l3888_388850


namespace solve_for_y_l3888_388820

theorem solve_for_y (x y : ℝ) (hx : x = 51) (heq : x^3 * y^2 - 4 * x^2 * y^2 + 4 * x * y^2 = 100800) :
  y = 1/34 ∨ y = -1/34 := by
  sorry

end solve_for_y_l3888_388820


namespace triangle_tangent_solution_l3888_388857

theorem triangle_tangent_solution (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z →
  4 * x * y * z = 4 * (x + y + z) →
  ∃ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ x = Real.tan A ∧ y = Real.tan B ∧ z = Real.tan C :=
by sorry

end triangle_tangent_solution_l3888_388857


namespace three_true_propositions_l3888_388839

theorem three_true_propositions
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_sum_prod : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) :
  ∃! (count : ℕ), count = 3 ∧ count = (
    (if b₂ < a₂ then 1 else 0) +
    (if a₃ < b₃ then 1 else 0) +
    (if a₁*a₂*a₃ < b₁*b₂*b₃ then 1 else 0) +
    (if (1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃) then 1 else 0)
  ) :=
sorry

end three_true_propositions_l3888_388839


namespace magnitude_of_complex_number_l3888_388890

theorem magnitude_of_complex_number (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 + 2*i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_complex_number_l3888_388890


namespace train_platform_length_l3888_388871

/-- The length of a train platform problem -/
theorem train_platform_length 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : train_length = 270)
  (h2 : platform1_length = 120)
  (h3 : time1 = 15)
  (h4 : time2 = 20) :
  ∃ (platform2_length : ℝ),
    platform2_length = 250 ∧ 
    (train_length + platform1_length) / time1 = 
    (train_length + platform2_length) / time2 := by
  sorry

end train_platform_length_l3888_388871


namespace unique_number_l3888_388865

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def begins_and_ends_with_2 (n : ℕ) : Prop :=
  n % 10 = 2 ∧ (n / 100000) % 10 = 2

def product_of_three_consecutive_even_integers (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)

theorem unique_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
             begins_and_ends_with_2 n ∧ 
             product_of_three_consecutive_even_integers n ∧
             n = 287232 :=
by sorry

end unique_number_l3888_388865


namespace volunteer_schedule_lcm_l3888_388808

theorem volunteer_schedule_lcm : Nat.lcm 2 (Nat.lcm 5 (Nat.lcm 9 11)) = 990 := by
  sorry

end volunteer_schedule_lcm_l3888_388808


namespace alberts_earnings_increase_l3888_388813

theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  1.27 * E = 567 →
  E + P * E = 562.54 →
  P = 0.26 := by
sorry

end alberts_earnings_increase_l3888_388813


namespace sandy_shopping_money_l3888_388896

theorem sandy_shopping_money (original_amount : ℝ) : 
  original_amount * 0.7 = 210 → original_amount = 300 :=
by sorry

end sandy_shopping_money_l3888_388896


namespace simplify_fraction_l3888_388863

theorem simplify_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (6 * a^2 * b * c) / (3 * a * b) = 2 * a * c := by
  sorry

end simplify_fraction_l3888_388863


namespace population_in_scientific_notation_l3888_388876

/-- Represents the population in billions -/
def population_in_billions : ℝ := 1.412

/-- Converts billions to scientific notation -/
def billions_to_scientific (x : ℝ) : ℝ := x * 10^9

/-- Theorem stating that 1.412 billion people in scientific notation is 1.412 × 10^9 -/
theorem population_in_scientific_notation :
  billions_to_scientific population_in_billions = 1.412 * 10^9 := by
  sorry

end population_in_scientific_notation_l3888_388876


namespace rows_remain_ascending_l3888_388894

/-- Represents a rectangular table of numbers -/
def Table (m n : ℕ) := Fin m → Fin n → ℝ

/-- Checks if a row is in ascending order -/
def isRowAscending (t : Table m n) (i : Fin m) : Prop :=
  ∀ j k : Fin n, j < k → t i j ≤ t i k

/-- Checks if a column is in ascending order -/
def isColumnAscending (t : Table m n) (j : Fin n) : Prop :=
  ∀ i k : Fin m, i < k → t i j ≤ t k j

/-- Sorts a row in ascending order -/
def sortRow (t : Table m n) (i : Fin m) : Table m n :=
  sorry

/-- Sorts a column in ascending order -/
def sortColumn (t : Table m n) (j : Fin n) : Table m n :=
  sorry

/-- Sorts all rows in ascending order -/
def sortAllRows (t : Table m n) : Table m n :=
  sorry

/-- Sorts all columns in ascending order -/
def sortAllColumns (t : Table m n) : Table m n :=
  sorry

/-- Main theorem: After sorting rows and then columns, rows remain in ascending order -/
theorem rows_remain_ascending (m n : ℕ) (t : Table m n) :
  ∀ i : Fin m, isRowAscending (sortAllColumns (sortAllRows t)) i :=
sorry

end rows_remain_ascending_l3888_388894


namespace inequality_solution_l3888_388824

theorem inequality_solution (x : ℝ) : 3 - 1 / (3 * x + 4) < 5 ↔ x > -3/2 ∧ 3 * x + 4 ≠ 0 := by
  sorry

end inequality_solution_l3888_388824


namespace max_correct_guesses_proof_l3888_388891

/-- Represents the maximum number of guaranteed correct hat color guesses 
    for n wise men with k insane among them. -/
def max_correct_guesses (n k : ℕ) : ℕ := n - k - 1

/-- Theorem stating that the maximum number of guaranteed correct hat color guesses
    is equal to n - k - 1, where n is the total number of wise men and k is the
    number of insane wise men. -/
theorem max_correct_guesses_proof (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 := by
  sorry

end max_correct_guesses_proof_l3888_388891


namespace sqrt_meaningful_range_l3888_388887

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 3 ≥ 0

-- Theorem statement
theorem sqrt_meaningful_range (x : ℝ) :
  meaningful_sqrt x ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l3888_388887


namespace gdp_scientific_notation_correct_l3888_388807

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℕ := 84300000

/-- The GDP expressed in scientific notation -/
def gdp_scientific : ScientificNotation where
  coefficient := 8.43
  exponent := 7
  is_valid := by sorry

/-- Theorem stating that the GDP value is correctly expressed in scientific notation -/
theorem gdp_scientific_notation_correct : 
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp := by sorry

end gdp_scientific_notation_correct_l3888_388807


namespace bus_speed_excluding_stoppages_l3888_388802

/-- Given a bus with an average speed including stoppages and the time it stops per hour,
    calculate the average speed excluding stoppages. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 20)
  (h2 : stop_time = 40) :
  speed_with_stops * (60 / (60 - stop_time)) = 60 :=
sorry

end bus_speed_excluding_stoppages_l3888_388802


namespace compound_vs_simple_interest_l3888_388898

/-- Calculate compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem compound_vs_simple_interest :
  ∀ P : ℝ,
  simple_interest P 0.1 2 = 600 →
  compound_interest P 0.1 2 = 630 := by
  sorry

end compound_vs_simple_interest_l3888_388898


namespace sqrt_3_minus_1_over_2_less_than_half_l3888_388849

theorem sqrt_3_minus_1_over_2_less_than_half : (Real.sqrt 3 - 1) / 2 < 1 / 2 := by
  sorry

end sqrt_3_minus_1_over_2_less_than_half_l3888_388849


namespace gcd_binomial_integer_l3888_388805

theorem gcd_binomial_integer (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ k : ℤ, (Nat.gcd m n : ℚ) / n * (n.choose m : ℚ) = k := by
  sorry

end gcd_binomial_integer_l3888_388805


namespace units_digit_G_500_l3888_388828

/-- The function G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G(500) is 0 -/
theorem units_digit_G_500 : unitsDigit (G 500) = 0 := by
  sorry

end units_digit_G_500_l3888_388828


namespace adrian_days_off_l3888_388800

/-- The number of days Adrian takes off per month for personal reasons -/
def personal_days_per_month : ℕ := 4

/-- The number of days Adrian takes off per month for professional development -/
def professional_days_per_month : ℕ := 2

/-- The number of days Adrian takes off per quarter for team-building events -/
def team_building_days_per_quarter : ℕ := 1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The total number of days Adrian takes off in a year -/
def total_days_off : ℕ :=
  personal_days_per_month * months_per_year +
  professional_days_per_month * months_per_year +
  team_building_days_per_quarter * quarters_per_year

theorem adrian_days_off : total_days_off = 76 := by
  sorry

end adrian_days_off_l3888_388800


namespace college_entrance_exam_score_l3888_388883

theorem college_entrance_exam_score
  (total_questions : ℕ)
  (answered_questions : ℕ)
  (raw_score : ℚ)
  (h1 : total_questions = 85)
  (h2 : answered_questions = 82)
  (h3 : raw_score = 67)
  (h4 : answered_questions ≤ total_questions) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ answered_questions ∧
    (correct_answers : ℚ) - 0.25 * ((answered_questions : ℚ) - (correct_answers : ℚ)) = raw_score ∧
    correct_answers = 69 :=
by sorry

end college_entrance_exam_score_l3888_388883


namespace min_value_theorem_l3888_388866

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | 2 * a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*x - 4*y + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2 * c - d + 2 = 0 → 1/c + 1/d ≥ 1/a + 1/b) ∧
  1/a + 1/b = 4 :=
sorry

end min_value_theorem_l3888_388866


namespace spoons_to_knives_ratio_l3888_388858

/-- Given a silverware set where the number of spoons is three times the number of knives,
    and the number of knives is 6, prove that the ratio of spoons to knives is 3:1. -/
theorem spoons_to_knives_ratio (knives : ℕ) (spoons : ℕ) : 
  knives = 6 → spoons = 3 * knives → spoons / knives = 3 :=
by
  sorry

#check spoons_to_knives_ratio

end spoons_to_knives_ratio_l3888_388858


namespace line_problem_l3888_388823

-- Define the lines
def l1 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l2 (m n x y : ℝ) : Prop := m*x + 4*y + n = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := m / 4 = 2

-- Define the distance between lines
def distance (m n : ℝ) : Prop := |2 + n/4| / Real.sqrt 5 = Real.sqrt 5

theorem line_problem (m n : ℝ) :
  parallel m → distance m n → (m + n = 36 ∨ m + n = -4) := by sorry

end line_problem_l3888_388823


namespace square_sum_given_square_sum_and_product_l3888_388847

theorem square_sum_given_square_sum_and_product
  (x y : ℝ) (h1 : (x + y)^2 = 25) (h2 : x * y = -6) :
  x^2 + y^2 = 37 := by
sorry

end square_sum_given_square_sum_and_product_l3888_388847


namespace angle_cosine_relation_l3888_388840

theorem angle_cosine_relation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  x / r = 1/4 ∧ y / r = 1/8 → z / r = Real.sqrt 59 / 8 := by
  sorry

end angle_cosine_relation_l3888_388840


namespace set_membership_l3888_388872

def M : Set ℤ := {a | ∃ b c : ℤ, a = b^2 - c^2}

theorem set_membership : (8 ∈ M) ∧ (9 ∈ M) ∧ (10 ∉ M) := by sorry

end set_membership_l3888_388872


namespace lost_people_problem_l3888_388818

/-- Calculates the number of people in the second group of lost people --/
def second_group_size (initial_group : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_food := initial_group * initial_days
  let remaining_food := total_food - (initial_group * days_passed)
  let total_people := remaining_food / remaining_days
  total_people - initial_group

/-- Theorem stating that given the problem conditions, the second group has 3 people --/
theorem lost_people_problem :
  second_group_size 9 5 1 3 = 3 := by
  sorry


end lost_people_problem_l3888_388818


namespace power_function_k_values_l3888_388882

/-- A function f(x) = ax^n is a power function if a ≠ 0 and n is a non-zero constant. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y = (k^2-k-5)x^2 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^2) → k = 3 ∨ k = -2 := by
  sorry


end power_function_k_values_l3888_388882


namespace jar_servings_calculation_l3888_388846

/-- Represents the contents and serving sizes of peanut butter and jelly in a jar -/
structure JarContents where
  pb_amount : ℚ  -- Amount of peanut butter in tablespoons
  jelly_amount : ℚ  -- Amount of jelly in tablespoons
  pb_serving : ℚ  -- Size of one peanut butter serving in tablespoons
  jelly_serving : ℚ  -- Size of one jelly serving in tablespoons

/-- Calculates the number of servings for peanut butter and jelly -/
def calculate_servings (jar : JarContents) : ℚ × ℚ :=
  (jar.pb_amount / jar.pb_serving, jar.jelly_amount / jar.jelly_serving)

/-- Theorem stating the correct number of servings for the given jar contents -/
theorem jar_servings_calculation (jar : JarContents)
  (h1 : jar.pb_amount = 35 + 2/3)
  (h2 : jar.jelly_amount = 18 + 1/2)
  (h3 : jar.pb_serving = 2 + 1/6)
  (h4 : jar.jelly_serving = 1) :
  calculate_servings jar = (16 + 18/39, 18 + 1/2) := by
  sorry

#eval calculate_servings {
  pb_amount := 35 + 2/3,
  jelly_amount := 18 + 1/2,
  pb_serving := 2 + 1/6,
  jelly_serving := 1
}

end jar_servings_calculation_l3888_388846
