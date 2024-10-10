import Mathlib

namespace prob_sum_three_is_half_l1863_186362

/-- A fair coin toss outcome -/
inductive CoinToss
  | heads
  | tails

/-- The numeric value associated with a coin toss -/
def coinValue (t : CoinToss) : ℕ :=
  match t with
  | CoinToss.heads => 1
  | CoinToss.tails => 2

/-- The sample space of two coin tosses -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.heads, CoinToss.heads),
   (CoinToss.heads, CoinToss.tails),
   (CoinToss.tails, CoinToss.heads),
   (CoinToss.tails, CoinToss.tails)]

/-- The event where the sum of two coin tosses is 3 -/
def sumThreeEvent (t : CoinToss × CoinToss) : Bool :=
  coinValue t.1 + coinValue t.2 = 3

/-- Theorem: The probability of obtaining a sum of 3 when tossing a fair coin twice is 1/2 -/
theorem prob_sum_three_is_half :
  (sampleSpace.filter sumThreeEvent).length / sampleSpace.length = 1 / 2 := by
  sorry

end prob_sum_three_is_half_l1863_186362


namespace root_product_of_quartic_l1863_186314

theorem root_product_of_quartic (a b c d : ℂ) : 
  (3 * a^4 - 8 * a^3 + a^2 + 4 * a - 12 = 0) ∧
  (3 * b^4 - 8 * b^3 + b^2 + 4 * b - 12 = 0) ∧
  (3 * c^4 - 8 * c^3 + c^2 + 4 * c - 12 = 0) ∧
  (3 * d^4 - 8 * d^3 + d^2 + 4 * d - 12 = 0) →
  a * b * c * d = -4 := by
sorry

end root_product_of_quartic_l1863_186314


namespace triangle_inequality_l1863_186358

/-- A triangle with sides x, y, and z satisfies the inequality
    (x+y+z)(x+y-z)(x+z-y)(z+y-x) ≤ 4x²y² -/
theorem triangle_inequality (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := by
  sorry

end triangle_inequality_l1863_186358


namespace max_value_parabola_l1863_186398

theorem max_value_parabola :
  (∀ x : ℝ, -x^2 + 5 ≤ 5) ∧ (∃ x : ℝ, -x^2 + 5 = 5) :=
by sorry

end max_value_parabola_l1863_186398


namespace quadratic_inequality_solution_condition_l1863_186327

theorem quadratic_inequality_solution_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by sorry

end quadratic_inequality_solution_condition_l1863_186327


namespace cabbage_area_is_one_sq_foot_l1863_186361

/-- Represents the cabbage garden problem --/
structure CabbageGarden where
  area_this_year : ℕ
  area_last_year : ℕ
  cabbages_this_year : ℕ
  cabbages_last_year : ℕ

/-- The area per cabbage is 1 square foot --/
theorem cabbage_area_is_one_sq_foot (garden : CabbageGarden)
  (h1 : garden.area_this_year = garden.cabbages_this_year)
  (h2 : garden.area_last_year = garden.cabbages_last_year)
  (h3 : garden.cabbages_this_year = 4096)
  (h4 : garden.cabbages_last_year = 3969)
  (h5 : ∃ n : ℕ, garden.area_this_year = n * n)
  (h6 : ∃ m : ℕ, garden.area_last_year = m * m) :
  garden.area_this_year / garden.cabbages_this_year = 1 := by
  sorry

#check cabbage_area_is_one_sq_foot

end cabbage_area_is_one_sq_foot_l1863_186361


namespace stating_failed_both_percentage_l1863_186397

/-- Represents the percentage of students in various categories -/
structure ExamResults where
  failed_hindi : ℝ
  failed_english : ℝ
  passed_both : ℝ

/-- 
Calculates the percentage of students who failed in both Hindi and English
given the exam results.
-/
def percentage_failed_both (results : ExamResults) : ℝ :=
  results.failed_hindi + results.failed_english - (100 - results.passed_both)

/-- 
Theorem stating that given the specific exam results, 
the percentage of students who failed in both subjects is 27%.
-/
theorem failed_both_percentage 
  (results : ExamResults)
  (h1 : results.failed_hindi = 25)
  (h2 : results.failed_english = 48)
  (h3 : results.passed_both = 54) :
  percentage_failed_both results = 27 := by
  sorry

#eval percentage_failed_both ⟨25, 48, 54⟩

end stating_failed_both_percentage_l1863_186397


namespace carpet_cost_is_576_l1863_186372

/-- The total cost of carpet squares needed to cover a rectangular floor -/
def total_carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_side_length * carpet_side_length
  let num_carpets := floor_area / carpet_area
  num_carpets * carpet_cost

/-- Proof that the total cost of carpet squares for the given floor is $576 -/
theorem carpet_cost_is_576 :
  total_carpet_cost 24 64 8 24 = 576 := by
  sorry

end carpet_cost_is_576_l1863_186372


namespace mans_rowing_speed_l1863_186364

theorem mans_rowing_speed (river_speed : ℝ) (round_trip_time : ℝ) (total_distance : ℝ) (still_water_speed : ℝ) : 
  river_speed = 2 →
  round_trip_time = 1 →
  total_distance = 5.333333333333333 →
  still_water_speed = 7.333333333333333 →
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed - river_speed ∧
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed + river_speed :=
by sorry

end mans_rowing_speed_l1863_186364


namespace binary_addition_subtraction_l1863_186322

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 1010₂ -/
def b1010 : List Bool := [true, false, true, false]

/-- Represents the binary number 1011₂ -/
def b1011 : List Bool := [true, false, true, true]

/-- Represents the binary number 11001₂ (the expected result) -/
def b11001 : List Bool := [true, true, false, false, true]

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b1010 + binary_to_nat b1011 =
  binary_to_nat b11001 := by
  sorry

end binary_addition_subtraction_l1863_186322


namespace farmer_apples_l1863_186367

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end farmer_apples_l1863_186367


namespace rectangle_square_area_ratio_l1863_186353

theorem rectangle_square_area_ratio : 
  let s : ℝ := 20
  let longer_side : ℝ := 1.05 * s
  let shorter_side : ℝ := 0.85 * s
  let area_R : ℝ := longer_side * shorter_side
  let area_S : ℝ := s * s
  area_R / area_S = 357 / 400 := by
sorry

end rectangle_square_area_ratio_l1863_186353


namespace coin_exchange_impossibility_l1863_186316

theorem coin_exchange_impossibility : ¬ ∃ n : ℕ, 1 + 4 * n = 26 := by
  sorry

end coin_exchange_impossibility_l1863_186316


namespace total_insects_eaten_l1863_186326

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by all animals -/
def total_insects : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_eaten :
  total_insects = 66 := by sorry

end total_insects_eaten_l1863_186326


namespace locus_equation_l1863_186305

def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (1, 0)

theorem locus_equation (x y : ℝ) :
  let M := (x, y)
  let dist_MA := Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2)
  let dist_MB := Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2)
  dist_MA = (1/2) * dist_MB → x^2 + y^2 = 4 := by
sorry

end locus_equation_l1863_186305


namespace regular_polygon_exterior_angle_18_l1863_186304

/-- A regular polygon with exterior angles each measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18 :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end regular_polygon_exterior_angle_18_l1863_186304


namespace total_height_increase_four_centuries_l1863_186331

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := 75

/-- Represents the number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- Theorem: The total increase in height over 4 centuries is 3000 meters -/
theorem total_height_increase_four_centuries : 
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = 3000 := by
  sorry

end total_height_increase_four_centuries_l1863_186331


namespace rectangular_field_area_l1863_186324

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio --/
theorem rectangular_field_area (perimeter : ℝ) (width_ratio : ℝ) : 
  perimeter = 72 ∧ width_ratio = 1/3 → 
  (perimeter / (2 * (1 + 1/width_ratio))) * (perimeter / (2 * (1 + 1/width_ratio))) / width_ratio = 243 := by
sorry

end rectangular_field_area_l1863_186324


namespace inverse_mod_31_l1863_186307

theorem inverse_mod_31 (h : (11⁻¹ : ZMod 31) = 3) : (20⁻¹ : ZMod 31) = 28 := by
  sorry

end inverse_mod_31_l1863_186307


namespace triangle_inequality_l1863_186351

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) := by
sorry

end triangle_inequality_l1863_186351


namespace village_population_problem_l1863_186306

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 3213 → P = 4200 := by
  sorry

end village_population_problem_l1863_186306


namespace sum_difference_remainder_l1863_186346

theorem sum_difference_remainder (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 3 * k + 1)
  (hc : ∃ k : ℤ, c = 3 * k - 1) :
  ∃ k : ℤ, a + b - c = 3 * k - 1 := by
sorry

end sum_difference_remainder_l1863_186346


namespace polygon_with_44_diagonals_has_11_sides_l1863_186312

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 44 diagonals has 11 sides -/
theorem polygon_with_44_diagonals_has_11_sides :
  ∃ (n : ℕ), n > 2 ∧ diagonals n = 44 → n = 11 := by
  sorry

end polygon_with_44_diagonals_has_11_sides_l1863_186312


namespace symmetry_implies_exponent_l1863_186341

theorem symmetry_implies_exponent (a b : ℝ) : 
  (2 * a + 1 = 1 ∧ -3 * a = -(3 - b)) → b^a = 1 := by
  sorry

end symmetry_implies_exponent_l1863_186341


namespace system_solution_l1863_186359

theorem system_solution (x y z : ℕ) : 
  x + y + z = 12 → 
  4 * x + 3 * y + 2 * z = 36 → 
  x ∈ ({0, 1, 2, 3, 4, 5, 6} : Set ℕ) := by
sorry

end system_solution_l1863_186359


namespace complement_M_intersect_N_l1863_186369

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.log (3 - x)}

-- Theorem statement
theorem complement_M_intersect_N : 
  (U \ M) ∩ N = {y | y ≤ 0} := by sorry

end complement_M_intersect_N_l1863_186369


namespace pen_cost_is_47_l1863_186375

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 47

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- Six pens and five pencils cost 380 cents -/
axiom condition1 : 6 * pen_cost + 5 * pencil_cost = 380

/-- Three pens and eight pencils cost 298 cents -/
axiom condition2 : 3 * pen_cost + 8 * pencil_cost = 298

/-- The cost of a pen is 47 cents -/
theorem pen_cost_is_47 : pen_cost = 47 := by sorry

end pen_cost_is_47_l1863_186375


namespace f_less_than_g_implies_a_bound_l1863_186338

open Real

/-- The function f parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

/-- The function g -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The theorem statement -/
theorem f_less_than_g_implies_a_bound 
  (h : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) : 
  a > log 2 - 1 := by
  sorry

end f_less_than_g_implies_a_bound_l1863_186338


namespace percentage_increase_l1863_186336

theorem percentage_increase (x y z : ℝ) (h1 : y = 0.5 * z) (h2 : x = 0.65 * z) :
  (x - y) / y * 100 = 30 := by
  sorry

end percentage_increase_l1863_186336


namespace pool_capacity_correct_l1863_186320

/-- The amount of water Grace's pool can contain -/
def pool_capacity : ℕ := 390

/-- The rate at which the first hose sprays water -/
def first_hose_rate : ℕ := 50

/-- The rate at which the second hose sprays water -/
def second_hose_rate : ℕ := 70

/-- The time the first hose runs alone -/
def first_hose_time : ℕ := 3

/-- The time both hoses run together -/
def both_hoses_time : ℕ := 2

/-- Theorem stating that the pool capacity is correct given the conditions -/
theorem pool_capacity_correct :
  pool_capacity = first_hose_rate * first_hose_time + 
    (first_hose_rate + second_hose_rate) * both_hoses_time :=
by sorry

end pool_capacity_correct_l1863_186320


namespace constant_function_shift_l1863_186377

/-- Given a function f that is constant 2 for all real numbers, 
    prove that f(x + 2) = 2 for all real numbers x. -/
theorem constant_function_shift (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 2) :
  ∀ x : ℝ, f (x + 2) = 2 := by
sorry

end constant_function_shift_l1863_186377


namespace centers_on_line_l1863_186380

-- Define the family of circles
def circle_family (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the line equation
def center_line (x y : ℝ) : Prop :=
  2*x - y - 5 = 0

-- Theorem statement
theorem centers_on_line :
  ∀ k : ℝ, k ≠ -1 →
  ∃ x y : ℝ, circle_family k x y ∧ center_line x y :=
sorry

end centers_on_line_l1863_186380


namespace sequence_general_term_l1863_186373

/-- The sequence defined by a₁ = -1 and aₙ₊₁ = 3aₙ - 1 has the general term aₙ = -(3ⁿ - 1)/2 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = -1) 
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 1) : 
    ∀ n : ℕ, n ≥ 1 → a n = -(3^n - 1) / 2 := by
  sorry

end sequence_general_term_l1863_186373


namespace triangle_problem_l1863_186384

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
def lawOfSines (t : Triangle) : Prop :=
  t.a / (Real.sin t.A) = t.b / (Real.sin t.B) ∧ 
  t.b / (Real.sin t.B) = t.c / (Real.sin t.C)

/-- The law of cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2*t.b*t.c*(Real.cos t.A) ∧
  t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*(Real.cos t.B) ∧
  t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*(Real.cos t.C)

theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 7)
  (h2 : t.c = 3)
  (h3 : Real.sin t.C / Real.sin t.B = 3/5)
  (h4 : lawOfSines t)
  (h5 : lawOfCosines t) :
  t.b = 5 ∧ Real.cos t.A = -1/2 := by
  sorry

end triangle_problem_l1863_186384


namespace quadratic_equations_common_root_l1863_186399

theorem quadratic_equations_common_root (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃! x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0)
  (h_common_root3 : ∃! x : ℝ, x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end quadratic_equations_common_root_l1863_186399


namespace octal_calculation_l1863_186311

/-- Converts a number from base 8 to base 10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two numbers in base 8 --/
def octal_multiply (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

/-- Subtracts two numbers in base 8 --/
def octal_subtract (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

theorem octal_calculation : 
  octal_subtract (octal_multiply 245 5) 107 = 1356 := by sorry

end octal_calculation_l1863_186311


namespace rooms_already_painted_l1863_186339

theorem rooms_already_painted
  (total_rooms : ℕ)
  (time_per_room : ℕ)
  (time_left : ℕ)
  (h1 : total_rooms = 10)
  (h2 : time_per_room = 8)
  (h3 : time_left = 16) :
  total_rooms - (time_left / time_per_room) = 8 :=
by sorry

end rooms_already_painted_l1863_186339


namespace prob_units_digit_8_is_3_16_l1863_186390

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

def units_digit (n : ℕ) : ℕ := n % 10

def prob_units_digit_8 : ℚ :=
  (Finset.filter (fun (a, b) => units_digit (3^a + 7^b) = 8) (Finset.product (Finset.range 100) (Finset.range 100))).card /
  (Finset.product (Finset.range 100) (Finset.range 100)).card

theorem prob_units_digit_8_is_3_16 : prob_units_digit_8 = 3 / 16 := by
  sorry

end prob_units_digit_8_is_3_16_l1863_186390


namespace layla_score_difference_l1863_186325

/-- Given Layla's score and the total score, calculate the difference between Layla's and Nahima's scores -/
def score_difference (layla_score : ℕ) (total_score : ℕ) : ℕ :=
  layla_score - (total_score - layla_score)

/-- Theorem: Given Layla's score of 70 and a total score of 112, Layla scored 28 more points than Nahima -/
theorem layla_score_difference :
  score_difference 70 112 = 28 := by
  sorry

#eval score_difference 70 112

end layla_score_difference_l1863_186325


namespace equation_roots_right_triangle_l1863_186302

-- Define the equation
def equation (x a b : ℝ) : Prop := |x^2 - 2*a*x + b| = 8

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2

-- Theorem statement
theorem equation_roots_right_triangle (a b : ℝ) :
  (∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation x a b ∧ equation y a b ∧ equation z a b ∧
    is_right_triangle x y z ∧
    (∀ w : ℝ, equation w a b → w = x ∨ w = y ∨ w = z)) →
  a + b = 264 :=
sorry

end equation_roots_right_triangle_l1863_186302


namespace simplify_expression_l1863_186340

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x^2)*(3*x^2) = 243*x^5 + 12*x^4 := by
  sorry

end simplify_expression_l1863_186340


namespace equation_identity_l1863_186335

theorem equation_identity (x : ℝ) : (3*x - 2)*(2*x + 5) - x = 6*x^2 + 2*(5*x - 5) := by
  sorry

end equation_identity_l1863_186335


namespace laptop_price_theorem_l1863_186317

/-- The sticker price of a laptop. -/
def stickerPrice : ℝ := 1100

/-- The price at store C after discount and rebate. -/
def storeCPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at store D after discount. -/
def storeDPrice (price : ℝ) : ℝ := 0.7 * price

theorem laptop_price_theorem : 
  storeCPrice stickerPrice = storeDPrice stickerPrice - 10 := by sorry

end laptop_price_theorem_l1863_186317


namespace solve_tank_problem_l1863_186344

def tank_problem (initial_capacity : ℝ) (leak_rate1 leak_rate2 fill_rate : ℝ)
  (leak_duration1 leak_duration2 fill_duration : ℝ) (missing_amount : ℝ) : Prop :=
  let total_loss := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_after_loss := initial_capacity - total_loss
  let current_amount := initial_capacity - missing_amount
  let amount_added := current_amount - remaining_after_loss
  fill_rate = amount_added / fill_duration

theorem solve_tank_problem :
  tank_problem 350000 32000 10000 40000 5 10 3 140000 := by
  sorry

end solve_tank_problem_l1863_186344


namespace fraction_simplification_l1863_186370

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / (x + y)) / (1 / (x - y)) = -1/9 := by sorry

end fraction_simplification_l1863_186370


namespace logic_statements_correctness_l1863_186318

theorem logic_statements_correctness :
  ∃! (n : Nat), n = 2 ∧
  (((∀ p q, p ∧ q → p ∨ q) ∧ (∃ p q, p ∨ q ∧ ¬(p ∧ q))) ∧
   ((∃ p q, ¬(p ∧ q) ∧ ¬(p ∨ q)) ∨ (∀ p q, p ∨ q → ¬(p ∧ q))) ∧
   ((∀ p q, ¬p → p ∨ q) ∧ (∃ p q, p ∨ q ∧ p)) ∧
   ((∀ p q, ¬p → ¬(p ∧ q)) ∧ (∃ p q, ¬(p ∧ q) ∧ p))) :=
by sorry

end logic_statements_correctness_l1863_186318


namespace wood_measurement_theorem_l1863_186350

/-- Represents the system of equations for the wood measurement problem -/
def wood_measurement_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - 1/2 * y = 1)

/-- Theorem stating that the given system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_theorem (x y : ℝ) :
  (∃ wood_length : ℝ, wood_length = x) →
  (∃ rope_length : ℝ, rope_length = y) →
  (y - x = 4.5) →
  (x - 1/2 * y = 1) →
  wood_measurement_equations x y :=
by
  sorry

end wood_measurement_theorem_l1863_186350


namespace max_value_of_f_l1863_186319

def f (x : ℝ) := 12 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c ∧ c = 9 :=
sorry

end max_value_of_f_l1863_186319


namespace packs_per_box_is_40_l1863_186385

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of packs in each box --/
def packs_per_box (business : DiaperBusiness) : ℕ :=
  (business.total_revenue / business.price_per_diaper) / 
  (business.diapers_per_pack * business.boxes_per_week)

/-- Theorem stating that the number of packs in each box is 40 --/
theorem packs_per_box_is_40 (business : DiaperBusiness) 
  (h1 : business.boxes_per_week = 30)
  (h2 : business.diapers_per_pack = 160)
  (h3 : business.price_per_diaper = 5)
  (h4 : business.total_revenue = 960000) :
  packs_per_box business = 40 := by
  sorry

end packs_per_box_is_40_l1863_186385


namespace logarithm_equality_l1863_186345

/-- Given the conditions on logarithms and the equation involving x^y, 
    prove that y equals 2q - p - r -/
theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) 
  (h1 : x ≠ 1)
  (h2 : Real.log a / p = Real.log b / q)
  (h3 : Real.log b / q = Real.log c / r)
  (h4 : Real.log b / q = Real.log x)
  (h5 : b^2 / (a * c) = x^y) :
  y = 2*q - p - r := by
  sorry

end logarithm_equality_l1863_186345


namespace angle_b_is_30_degrees_l1863_186323

-- Define the structure of our triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles of the triangle
  (white_angle gray_angle : ℝ)  -- Measures of white and gray angles
  (b : ℝ)  -- The angle we want to determine

-- State the theorem
theorem angle_b_is_30_degrees (t : Triangle) : 
  t.A = 60 ∧  -- Given angle is 60°
  t.A + t.B + t.C = 180 ∧  -- Sum of angles in a triangle is 180°
  t.A + 2 * t.gray_angle + (180 - 2 * t.white_angle) = 180 ∧  -- Equation for triangle ABC
  t.gray_angle + t.b + (180 - 2 * t.white_angle) = 180  -- Equation for triangle BCD
  → t.b = 30 := by
sorry  -- Proof is omitted as per instructions

end angle_b_is_30_degrees_l1863_186323


namespace work_earnings_equality_l1863_186387

theorem work_earnings_equality (t : ℚ) : 
  (t + 2) * (4*t - 2) = (4*t - 7) * (t + 1) + 4 → t = 1/9 := by
  sorry

end work_earnings_equality_l1863_186387


namespace max_sphere_radius_squared_in_cones_l1863_186337

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ :=
  sorry

/-- Theorem stating the maximum squared radius of a sphere in the given configuration -/
theorem max_sphere_radius_squared_in_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSphereRadiusSquared ic = 144 / 29 := by
  sorry

end max_sphere_radius_squared_in_cones_l1863_186337


namespace min_reciprocal_sum_l1863_186349

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 1) :
  (1 / x + 1 / y) ≥ 9 :=
by sorry

end min_reciprocal_sum_l1863_186349


namespace ratio_equality_l1863_186391

theorem ratio_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : (x + z) / (2 * z - x) = x / y)
  (h_eq2 : (z + 2 * y) / (2 * x - z) = x / y) : 
  x / y = 2 := by
sorry

end ratio_equality_l1863_186391


namespace simplify_expression_l1863_186360

theorem simplify_expression (x y : ℝ) : (3 * x + 22) + (150 * y + 22) = 3 * x + 150 * y + 44 := by
  sorry

end simplify_expression_l1863_186360


namespace instantaneous_velocity_at_t_1_l1863_186383

-- Define the position function S(t)
def S (t : ℝ) : ℝ := 2 * t^2 + t

-- Define the velocity function as the derivative of S(t)
def v (t : ℝ) : ℝ := 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_t_1 : v 1 = 5 := by
  sorry

end instantaneous_velocity_at_t_1_l1863_186383


namespace unique_value_sum_l1863_186363

/-- Given that {a, b, c} = {0, 1, 2} and exactly one of (a ≠ 2), (b = 2), (c ≠ 0) is true,
    prove that a + 2b + 5c = 7 -/
theorem unique_value_sum (a b c : ℤ) : 
  ({a, b, c} : Set ℤ) = {0, 1, 2} →
  ((a ≠ 2) ∨ (b = 2) ∨ (c ≠ 0)) ∧
  (¬((a ≠ 2) ∧ (b = 2)) ∧ ¬((a ≠ 2) ∧ (c ≠ 0)) ∧ ¬((b = 2) ∧ (c ≠ 0))) →
  a + 2*b + 5*c = 7 := by
  sorry

end unique_value_sum_l1863_186363


namespace store_price_calculation_l1863_186347

/-- If an item's online price is 300 yuan and it's 20% less than the store price,
    then the store price is 375 yuan. -/
theorem store_price_calculation (online_price store_price : ℝ) : 
  online_price = 300 →
  online_price = store_price - 0.2 * store_price →
  store_price = 375 := by
  sorry

end store_price_calculation_l1863_186347


namespace profit_percentage_calculation_l1863_186379

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 620 →
  selling_price = 775 →
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_calculation_l1863_186379


namespace sum_and_double_l1863_186308

theorem sum_and_double : (2345 + 3452 + 4523 + 5234) * 2 = 31108 := by
  sorry

end sum_and_double_l1863_186308


namespace hyperbola_center_l1863_186328

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola_equation x y → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

theorem hyperbola_center : is_center 2 4 := by
  sorry

end hyperbola_center_l1863_186328


namespace earthwork_inequality_l1863_186352

/-- Proves the inequality for the required average daily earthwork to complete the project ahead of schedule. -/
theorem earthwork_inequality (total : ℝ) (days : ℕ) (first_day : ℝ) (ahead : ℕ) (x : ℝ) 
  (h_total : total = 300)
  (h_days : days = 6)
  (h_first_day : first_day = 60)
  (h_ahead : ahead = 2)
  : 3 * x ≥ total - first_day :=
by
  sorry

#check earthwork_inequality

end earthwork_inequality_l1863_186352


namespace dinosaur_book_cost_l1863_186386

/-- The cost of a dinosaur book, given the total cost of three books and the costs of two of them. -/
theorem dinosaur_book_cost (total_cost dictionary_cost cookbook_cost : ℕ) 
  (h_total : total_cost = 37)
  (h_dict : dictionary_cost = 11)
  (h_cook : cookbook_cost = 7) :
  total_cost - dictionary_cost - cookbook_cost = 19 := by
  sorry

end dinosaur_book_cost_l1863_186386


namespace arithmetic_sequence_ratio_l1863_186356

/-- Sums of arithmetic sequences -/
def S (n : ℕ) : ℝ := sorry

/-- Sums of arithmetic sequences -/
def T (n : ℕ) : ℝ := sorry

/-- Terms of the first arithmetic sequence -/
def a : ℕ → ℝ := sorry

/-- Terms of the second arithmetic sequence -/
def b : ℕ → ℝ := sorry

theorem arithmetic_sequence_ratio :
  (∀ n : ℕ+, S n / T n = n / (2 * n + 1)) →
  a 6 / b 6 = 11 / 23 := by sorry

end arithmetic_sequence_ratio_l1863_186356


namespace double_root_condition_l1863_186366

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is double the other, 
then 2b^2 = 9ac.
-/
theorem double_root_condition (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₂ = 2 * x₁ → 
  2 * b^2 = 9 * a * c := by
sorry

end double_root_condition_l1863_186366


namespace quadratic_equation_conversion_quadratic_coefficients_l1863_186365

theorem quadratic_equation_conversion (x : ℝ) : 
  (x^2 - 8*x = 10) ↔ (x^2 - 8*x - 10 = 0) :=
by sorry

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 8*x - 10 = 0 ↔ a*x^2 + b*x + c = 0) ∧ 
  a = 1 ∧ b = -8 ∧ c = -10 :=
by sorry

end quadratic_equation_conversion_quadratic_coefficients_l1863_186365


namespace bank_deposit_problem_l1863_186310

/-- Calculates the total amount after maturity for a fixed-term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

theorem bank_deposit_problem :
  let principal : ℝ := 100000
  let rate : ℝ := 0.0315
  let time : ℝ := 2
  totalAmount principal rate time = 106300 := by
  sorry

end bank_deposit_problem_l1863_186310


namespace otimes_four_eight_l1863_186389

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := a / b + b / a

-- Theorem statement
theorem otimes_four_eight : otimes 4 8 = 5/2 := by
  sorry

end otimes_four_eight_l1863_186389


namespace max_value_d_l1863_186303

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_d_l1863_186303


namespace triangle_inequality_l1863_186388

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end triangle_inequality_l1863_186388


namespace existence_of_complementary_sequences_l1863_186329

def s (x y : ℝ) : Set ℕ := {s | ∃ n : ℕ, s = ⌊n * x + y⌋}

theorem existence_of_complementary_sequences (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (s r 0 ∩ s u v = ∅) ∧ (s r 0 ∪ s u v = Set.univ) := by
  sorry

end existence_of_complementary_sequences_l1863_186329


namespace f_mono_increasing_condition_l1863_186309

/-- A quadratic function f(x) = ax^2 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

/-- The property of being monotonically increasing on (0, +∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

/-- The condition a ≥ 0 is sufficient but not necessary for f to be monotonically increasing on (0, +∞) -/
theorem f_mono_increasing_condition (a : ℝ) :
  (a ≥ 0 → MonoIncreasing (f a)) ∧
  ¬(MonoIncreasing (f a) → a ≥ 0) :=
sorry

end f_mono_increasing_condition_l1863_186309


namespace cos_2x_at_min_y_l1863_186330

theorem cos_2x_at_min_y (x : ℝ) : 
  let y := 2 * (Real.sin x)^6 + (Real.cos x)^6
  (∀ z : ℝ, y ≤ 2 * (Real.sin z)^6 + (Real.cos z)^6) →
  Real.cos (2 * x) = 3 - 2 * Real.sqrt 2 := by
sorry

end cos_2x_at_min_y_l1863_186330


namespace hypotenuse_length_of_special_triangle_l1863_186381

/-- Given a right-angled triangle with side lengths a, b, and c (where c is the hypotenuse),
    if the sum of squares of all sides is 2000 and the perimeter is 60,
    then the hypotenuse length is 10√10. -/
theorem hypotenuse_length_of_special_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  a + b + c = 60 →
  c = 10 * Real.sqrt 10 := by
  sorry

end hypotenuse_length_of_special_triangle_l1863_186381


namespace unique_multiplication_solution_l1863_186354

/-- Represents a three-digit number in the form abb --/
def three_digit (a b : Nat) : Nat := 100 * a + 10 * b + b

/-- Represents a four-digit number in the form bcb1 --/
def four_digit (b c : Nat) : Nat := 1000 * b + 100 * c + 10 * b + 1

theorem unique_multiplication_solution :
  ∃! (a b c : Nat),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    three_digit a b * c = four_digit b c ∧
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end unique_multiplication_solution_l1863_186354


namespace intersection_nonempty_a_subset_b_l1863_186300

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / (x - 3) < -1}
def B (a : ℝ) : Set ℝ := {x | (x - (a^2 + 2)) / (x - a) < 0}

-- Part 1: Intersection is non-empty
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ a < 0 ∨ (0 < a ∧ a < 3) :=
sorry

-- Part 2: A is a subset of B
theorem a_subset_b (a : ℝ) :
  A ⊆ B a ↔ a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end intersection_nonempty_a_subset_b_l1863_186300


namespace first_company_daily_rate_l1863_186313

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48.0

theorem first_company_daily_rate :
  first_company_rate + first_company_per_mile * miles_driven =
  city_rentals_rate + city_rentals_per_mile * miles_driven :=
by sorry

end first_company_daily_rate_l1863_186313


namespace equation_solutions_l1863_186374

theorem equation_solutions : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ = (-1 + Real.sqrt 6) / 5 ∧ 
     x₂ = (-1 - Real.sqrt 6) / 5 ∧ 
     5 * x₁^2 + 2 * x₁ - 1 = 0 ∧ 
     5 * x₂^2 + 2 * x₂ - 1 = 0) ∧
    (x₃ = 3 ∧ 
     x₄ = -4 ∧ 
     x₃ * (x₃ - 3) - 4 * (3 - x₃) = 0 ∧ 
     x₄ * (x₄ - 3) - 4 * (3 - x₄) = 0) := by
  sorry

end equation_solutions_l1863_186374


namespace three_person_subcommittees_from_seven_l1863_186334

theorem three_person_subcommittees_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  Nat.choose n k = 35 := by
  sorry

end three_person_subcommittees_from_seven_l1863_186334


namespace box_value_l1863_186315

theorem box_value (x : ℝ) : x * (-2) = 4 → x = -2 := by
  sorry

end box_value_l1863_186315


namespace smallest_fraction_greater_than_five_sixths_l1863_186343

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_five_sixths_l1863_186343


namespace derivative_of_f_l1863_186378

noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 8 * Real.pi^2 * x := by sorry

end derivative_of_f_l1863_186378


namespace banana_ratio_l1863_186382

/-- Theorem about the ratio of bananas in Raj's basket to bananas eaten -/
theorem banana_ratio (initial_bananas : ℕ) (bananas_left_on_tree : ℕ) (bananas_eaten : ℕ) :
  initial_bananas = 310 →
  bananas_left_on_tree = 100 →
  bananas_eaten = 70 →
  (initial_bananas - bananas_left_on_tree - bananas_eaten) / bananas_eaten = 2 := by
  sorry


end banana_ratio_l1863_186382


namespace trig_function_amplitude_l1863_186394

theorem trig_function_amplitude 
  (y : ℝ → ℝ) 
  (a b c d : ℝ) 
  (h1 : ∀ x, y x = a * Real.cos (b * x + c) + d) 
  (h2 : ∃ x, y x = 4) 
  (h3 : ∃ x, y x = 0) 
  (h4 : ∀ x, y x ≤ 4) 
  (h5 : ∀ x, y x ≥ 0) : 
  a = 2 := by sorry

end trig_function_amplitude_l1863_186394


namespace geometric_sequence_a11_l1863_186371

/-- A geometric sequence with a_3 = 3 and a_7 = 6 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 3 = 3 ∧ 
  a 7 = 6

theorem geometric_sequence_a11 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 11 = 12 := by
  sorry

end geometric_sequence_a11_l1863_186371


namespace triangle_problem_l1863_186321

theorem triangle_problem (a b c A B C : ℝ) : 
  (2 * c - 2 * a * Real.cos B = b) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 4) →
  (c^2 + a * b * Real.cos C + a^2 = 4) →
  (A = π/3 ∧ a = Real.sqrt 7 / 2) := by
  sorry

end triangle_problem_l1863_186321


namespace investment_interest_rates_l1863_186332

theorem investment_interest_rates 
  (P1 P2 : ℝ) 
  (r1 r2 r3 r4 r5 : ℝ) :
  P1 / P2 = 2 / 3 →
  P1 * 5 * 8 / 100 = 840 →
  P2 * (r1 + r2 + r3 + r4 + r5) / 100 = 840 →
  r1 + r2 + r3 + r4 + r5 = 26.67 :=
by sorry

end investment_interest_rates_l1863_186332


namespace patricia_books_count_l1863_186368

/-- Given the number of books read by Candice, calculate the number of books read by Patricia -/
def books_read_by_patricia (candice_books : ℕ) : ℕ :=
  let amanda_books := candice_books / 3
  let kara_books := amanda_books / 2
  7 * kara_books

/-- Theorem stating that if Candice read 18 books, Patricia read 21 books -/
theorem patricia_books_count (h : books_read_by_patricia 18 = 21) : 
  books_read_by_patricia 18 = 21 := by
  sorry

#eval books_read_by_patricia 18

end patricia_books_count_l1863_186368


namespace tangent_slope_of_circle_l1863_186348

/-- Given a circle with center (1,3) and a point (4,7) on the circle,
    the slope of the line tangent to the circle at (4,7) is -3/4 -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (1, 3) →
  point = (4, 7) →
  (let slope_tangent := -(((point.2 - center.2) / (point.1 - center.1))⁻¹)
   slope_tangent = -3/4) :=
by sorry

end tangent_slope_of_circle_l1863_186348


namespace workday_meeting_percentage_l1863_186376

def workday_hours : ℕ := 8
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 40
def third_meeting_duration : ℕ := 30
def overlap_duration : ℕ := 10

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour

def second_meeting_duration : ℕ := 2 * first_meeting_duration

def effective_second_meeting_duration : ℕ := second_meeting_duration - overlap_duration

def total_meeting_time : ℕ := first_meeting_duration + effective_second_meeting_duration + third_meeting_duration

def meeting_percentage : ℚ := (total_meeting_time : ℚ) / (total_workday_minutes : ℚ) * 100

theorem workday_meeting_percentage :
  ∃ (x : ℚ), abs (meeting_percentage - x) < 1 ∧ ⌊x⌋ = 29 := by sorry

end workday_meeting_percentage_l1863_186376


namespace road_trip_gas_cost_l1863_186393

/-- Calculates the total cost of filling a car's gas tank at multiple stations -/
def total_gas_cost (tank_capacity : ℝ) (gas_prices : List ℝ) : ℝ :=
  (gas_prices.map (· * tank_capacity)).sum

/-- Proves that the total cost of filling a 12-gallon tank at 4 stations with given prices is $180 -/
theorem road_trip_gas_cost : 
  let tank_capacity : ℝ := 12
  let gas_prices : List ℝ := [3, 3.5, 4, 4.5]
  total_gas_cost tank_capacity gas_prices = 180 := by
  sorry

#eval total_gas_cost 12 [3, 3.5, 4, 4.5]

end road_trip_gas_cost_l1863_186393


namespace distance_between_points_l1863_186342

def point1 : ℝ × ℝ := (-5, 3)
def point2 : ℝ × ℝ := (6, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 265 := by
  sorry

end distance_between_points_l1863_186342


namespace total_beignets_in_16_weeks_l1863_186392

/-- The number of beignets Sandra eats each morning -/
def daily_beignets : ℕ := 3

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating the total number of beignets Sandra will eat in 16 weeks -/
theorem total_beignets_in_16_weeks : 
  daily_beignets * days_per_week * weeks = 336 := by
  sorry

end total_beignets_in_16_weeks_l1863_186392


namespace isosceles_triangle_base_angle_l1863_186357

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The base angle is either 70° or 55°
  (α = 70 ∨ α = 55 ∨ β = 70 ∨ β = 55 ∨ γ = 70 ∨ γ = 55) :=
by sorry

end isosceles_triangle_base_angle_l1863_186357


namespace boats_first_meeting_distance_l1863_186395

/-- Two boats traveling across a river, meeting twice without stopping at shores -/
structure BoatMeeting where
  /-- Total distance between shore A and B in yards -/
  total_distance : ℝ
  /-- Distance from shore B to the second meeting point in yards -/
  second_meeting_distance : ℝ
  /-- Distance from shore A to the first meeting point in yards -/
  first_meeting_distance : ℝ

/-- Theorem stating that the boats first meet at 300 yards from shore A -/
theorem boats_first_meeting_distance (meeting : BoatMeeting)
    (h1 : meeting.total_distance = 1200)
    (h2 : meeting.second_meeting_distance = 300) :
    meeting.first_meeting_distance = 300 := by
  sorry


end boats_first_meeting_distance_l1863_186395


namespace distance_traveled_l1863_186333

/-- Represents the actual distance traveled in kilometers -/
def actual_distance : ℝ := 33.75

/-- Represents the initial walking speed in km/hr -/
def initial_speed : ℝ := 15

/-- Represents the faster walking speed in km/hr -/
def faster_speed : ℝ := 35

/-- Represents the fraction of the distance that is uphill -/
def uphill_fraction : ℝ := 0.6

/-- Represents the decrease in speed for uphill portion -/
def uphill_speed_decrease : ℝ := 0.1

/-- Represents the additional distance covered at faster speed -/
def additional_distance : ℝ := 45

theorem distance_traveled :
  ∃ (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance * uphill_fraction = (faster_speed * (1 - uphill_speed_decrease)) * (time * uphill_fraction) ∧
    actual_distance * (1 - uphill_fraction) = faster_speed * (time * (1 - uphill_fraction)) :=
by sorry

end distance_traveled_l1863_186333


namespace typing_service_problem_l1863_186301

/-- Represents the typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (total_cost : ℕ) 
  (first_typing_cost : ℕ) 
  (revision_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_revised_twice = 30)
  (h3 : total_cost = 1400)
  (h4 : first_typing_cost = 10)
  (h5 : revision_cost = 5) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 20 ∧
    total_cost = 
      first_typing_cost * total_pages + 
      revision_cost * pages_revised_once + 
      2 * revision_cost * pages_revised_twice :=
by sorry

end typing_service_problem_l1863_186301


namespace complex_first_quadrant_l1863_186355

theorem complex_first_quadrant (a : ℝ) : 
  (∃ (z : ℂ), z = (1 : ℂ) / (1 + a * Complex.I) ∧ z.re > 0 ∧ z.im > 0) ↔ a < 0 :=
sorry

end complex_first_quadrant_l1863_186355


namespace gnomes_in_fifth_house_l1863_186396

theorem gnomes_in_fifth_house 
  (total_houses : ℕ)
  (gnomes_per_house : ℕ)
  (houses_with_known_gnomes : ℕ)
  (total_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : gnomes_per_house = 3)
  (h3 : houses_with_known_gnomes = 4)
  (h4 : total_gnomes = 20) :
  total_gnomes - (houses_with_known_gnomes * gnomes_per_house) = 8 :=
by
  sorry

#check gnomes_in_fifth_house

end gnomes_in_fifth_house_l1863_186396
