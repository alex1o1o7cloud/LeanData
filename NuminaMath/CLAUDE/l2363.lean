import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2363_236380

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a > 0) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2363_236380


namespace NUMINAMATH_CALUDE_circle_area_difference_l2363_236326

theorem circle_area_difference : ∀ (π : ℝ), 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * (d2/2)^2
  area1 - area2 = 675 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2363_236326


namespace NUMINAMATH_CALUDE_max_third_term_l2363_236396

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term :
  ∀ seq : ArithSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithSequence, third_term seq = 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_l2363_236396


namespace NUMINAMATH_CALUDE_roots_sum_product_l2363_236320

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 4*a - 1 = 0) → 
  (b^4 - 4*b - 1 = 0) → 
  (∀ x : ℝ, x ≠ a ∧ x ≠ b → x^4 - 4*x - 1 ≠ 0) →
  a * b + a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_l2363_236320


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l2363_236307

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_size : ℕ) : ℕ :=
  (length * width * height) / (cube_size ^ 3)

/-- Theorem: The minimum number of 3 cm³ cubes required to build a 9 cm × 12 cm × 3 cm box is 108 -/
theorem min_cubes_for_specific_box :
  min_cubes 9 12 3 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l2363_236307


namespace NUMINAMATH_CALUDE_color_tv_price_l2363_236375

/-- The original price of a color TV before price changes --/
def original_price : ℝ := 2250

/-- The price increase percentage --/
def price_increase : ℝ := 0.4

/-- The discount percentage --/
def discount : ℝ := 0.2

/-- The additional profit per TV --/
def additional_profit : ℝ := 270

theorem color_tv_price : 
  (original_price * (1 + price_increase) * (1 - discount)) - original_price = additional_profit :=
by sorry

end NUMINAMATH_CALUDE_color_tv_price_l2363_236375


namespace NUMINAMATH_CALUDE_expensive_handcuffs_time_l2363_236338

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def time_expensive : ℝ := 8

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def time_cheap : ℝ := 6

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_time : ℝ := 42

theorem expensive_handcuffs_time :
  time_expensive = (total_time - num_friends * time_cheap) / num_friends := by
  sorry

end NUMINAMATH_CALUDE_expensive_handcuffs_time_l2363_236338


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2363_236392

theorem no_integer_solutions : ¬∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2363_236392


namespace NUMINAMATH_CALUDE_continuous_injective_on_irrationals_implies_injective_monotonic_l2363_236316

/-- A function is injective on irrational numbers -/
def InjectiveOnIrrationals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Irrational x → Irrational y → x ≠ y → f x ≠ f y

/-- A function is strictly monotonic -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ ∀ x y : ℝ, x < y → f x > f y

theorem continuous_injective_on_irrationals_implies_injective_monotonic
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_inj_irr : InjectiveOnIrrationals f) :
  Function.Injective f ∧ StrictlyMonotonic f :=
sorry

end NUMINAMATH_CALUDE_continuous_injective_on_irrationals_implies_injective_monotonic_l2363_236316


namespace NUMINAMATH_CALUDE_number_of_papers_l2363_236348

/-- Represents the marks obtained in each paper -/
structure PaperMarks where
  fullMarks : ℝ
  proportions : List ℝ
  totalPapers : ℕ

/-- Checks if the given PaperMarks satisfies the problem conditions -/
def satisfiesConditions (pm : PaperMarks) : Prop :=
  pm.proportions = [5, 6, 7, 8, 9] ∧
  pm.totalPapers = pm.proportions.length ∧
  (pm.proportions.sum * pm.fullMarks * 0.6 = pm.proportions.sum * pm.fullMarks) ∧
  (List.filter (fun p => p * pm.fullMarks > 0.5 * pm.fullMarks) pm.proportions).length = 5

/-- Theorem stating that if the conditions are satisfied, the number of papers is 5 -/
theorem number_of_papers (pm : PaperMarks) (h : satisfiesConditions pm) : pm.totalPapers = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_papers_l2363_236348


namespace NUMINAMATH_CALUDE_max_gcd_11n_plus_3_6n_plus_1_l2363_236373

theorem max_gcd_11n_plus_3_6n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ gcd (11 * k + 3) (6 * k + 1) = 7 ∧
  ∀ (n : ℕ), n > 0 → gcd (11 * n + 3) (6 * n + 1) ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_11n_plus_3_6n_plus_1_l2363_236373


namespace NUMINAMATH_CALUDE_basketball_scores_l2363_236319

def first_ten_games : List Nat := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

theorem basketball_scores (game_11 game_12 : Nat) : 
  game_11 < 10 →
  game_12 < 10 →
  (List.sum first_ten_games + game_11) % 11 = 0 →
  (List.sum first_ten_games + game_11 + game_12) % 12 = 0 →
  game_11 * game_12 = 0 := by
sorry

end NUMINAMATH_CALUDE_basketball_scores_l2363_236319


namespace NUMINAMATH_CALUDE_sum_of_digits_5mul_permutation_l2363_236351

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if two natural numbers are permutations of each other's digits -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Theorem: If A is a permutation of B's digits, then sum of digits of 5A equals sum of digits of 5B -/
theorem sum_of_digits_5mul_permutation (A B : ℕ) :
  isDigitPermutation A B → sumOfDigits (5 * A) = sumOfDigits (5 * B) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_5mul_permutation_l2363_236351


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2363_236374

/-- Proves that a cycle sold at a 25% loss for 1050 had an original price of 1400 -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1050)
  (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2363_236374


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l2363_236331

-- Define the triangle ABC and related points
variable (A B C D E F : ℝ × ℝ)
variable (α : ℝ)

-- Define the conditions
axiom right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
axiom parallel_line : (D.2 - E.2) / (D.1 - E.1) = (B.2 - A.2) / (B.1 - A.1)
axiom DE_length : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 2
axiom BE_length : Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 1
axiom BF_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 1
axiom F_on_hypotenuse : (F.1 - A.1) / (B.1 - A.1) = (F.2 - A.2) / (B.2 - A.2)
axiom angle_FCB : Real.cos α = (F.1 - C.1) / Real.sqrt ((F.1 - C.1)^2 + (F.2 - C.2)^2)

-- Define the theorem
theorem area_of_triangle_ABC :
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) =
  (1/2) * (2 * Real.cos (2*α) + 1)^2 * Real.tan (2*α) := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l2363_236331


namespace NUMINAMATH_CALUDE_certain_number_minus_32_l2363_236313

theorem certain_number_minus_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_32_l2363_236313


namespace NUMINAMATH_CALUDE_complex_power_sum_l2363_236386

theorem complex_power_sum (w : ℂ) (h : w + 1 / w = 2 * Real.cos (5 * π / 180)) :
  w^1000 + 1 / w^1000 = -(Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2363_236386


namespace NUMINAMATH_CALUDE_water_formed_is_zero_l2363_236394

-- Define the chemical compounds
inductive Compound
| NH4Cl
| NaOH
| BaNO3_2
| NH4OH
| NaCl
| HNO3
| NH4NO3
| H2O
| NaNO3
| BaCl2

-- Define a reaction
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

-- Define the given reactions
def reaction1 : Reaction :=
{ reactants := [(Compound.NH4Cl, 1), (Compound.NaOH, 1)]
, products := [(Compound.NH4OH, 1), (Compound.NaCl, 1)] }

def reaction2 : Reaction :=
{ reactants := [(Compound.NH4OH, 1), (Compound.HNO3, 1)]
, products := [(Compound.NH4NO3, 1), (Compound.H2O, 1)] }

def reaction3 : Reaction :=
{ reactants := [(Compound.BaNO3_2, 1), (Compound.NaCl, 2)]
, products := [(Compound.NaNO3, 2), (Compound.BaCl2, 1)] }

-- Define the initial reactants
def initialReactants : List (Compound × ℕ) :=
[(Compound.NH4Cl, 3), (Compound.NaOH, 3), (Compound.BaNO3_2, 2)]

-- Define a function to calculate the moles of water formed
def molesOfWaterFormed (initialReactants : List (Compound × ℕ)) 
                       (reactions : List Reaction) : ℕ :=
  sorry

-- Theorem statement
theorem water_formed_is_zero :
  molesOfWaterFormed initialReactants [reaction1, reaction2, reaction3] = 0 :=
sorry

end NUMINAMATH_CALUDE_water_formed_is_zero_l2363_236394


namespace NUMINAMATH_CALUDE_set_operations_l2363_236333

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x : ℝ | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2363_236333


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2363_236323

/-- The line equation passing through a fixed point for any real k -/
def line_equation (k x y : ℝ) : ℝ := (2*k - 1)*x - (k + 3)*y - (k - 11)

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2363_236323


namespace NUMINAMATH_CALUDE_log_identities_l2363_236339

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_identities (a P : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  (log (a^2) P = (log a P) / 2) ∧
  (log (Real.sqrt a) P = 2 * log a P) ∧
  (log (1/a) P = -(log a P)) := by
  sorry

end NUMINAMATH_CALUDE_log_identities_l2363_236339


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2363_236346

theorem quadratic_factorization :
  ∀ x : ℝ, 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2363_236346


namespace NUMINAMATH_CALUDE_tameka_cracker_sales_l2363_236312

/-- Proves that given the conditions in the problem, Tameka sold 30 more boxes on Saturday than on Friday --/
theorem tameka_cracker_sales : ∀ (saturday_sales : ℕ),
  (40 + saturday_sales + saturday_sales / 2 = 145) →
  (saturday_sales = 40 + 30) := by
  sorry

end NUMINAMATH_CALUDE_tameka_cracker_sales_l2363_236312


namespace NUMINAMATH_CALUDE_rational_function_value_l2363_236372

/-- A rational function with specific properties -/
structure RationalFunction where
  r : ℝ → ℝ
  s : ℝ → ℝ
  r_linear : ∃ a b : ℝ, ∀ x, r x = a * x + b
  s_quadratic : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  asymptote_neg_two : s (-2) = 0
  asymptote_three : s 3 = 0
  passes_origin : r 0 = 0 ∧ s 0 ≠ 0
  passes_one_neg_two : r 1 / s 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.r 2 / f.s 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l2363_236372


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l2363_236390

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  let newHour := (totalMinutes / 60) % 24
  let newMinute := totalMinutes % 60
  ⟨newHour, newMinute, by sorry, by sorry⟩

theorem dinner_time_calculation (start : Time) 
    (h_start : start = ⟨16, 0, by sorry, by sorry⟩)
    (commute : Nat) (h_commute : commute = 30)
    (grocery : Nat) (h_grocery : grocery = 30)
    (drycleaning : Nat) (h_drycleaning : drycleaning = 10)
    (dog : Nat) (h_dog : dog = 20)
    (cooking : Nat) (h_cooking : cooking = 90) :
  addMinutes start (commute + grocery + drycleaning + dog + cooking) = ⟨19, 0, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_calculation_l2363_236390


namespace NUMINAMATH_CALUDE_courtney_marble_weight_l2363_236364

/-- The weight of Courtney's marble collection --/
def marbleCollectionWeight (firstJarCount : ℕ) (firstJarWeight : ℚ) 
  (secondJarWeight : ℚ) (thirdJarWeight : ℚ) : ℚ :=
  firstJarCount * firstJarWeight + 
  (2 * firstJarCount) * secondJarWeight + 
  (firstJarCount / 4) * thirdJarWeight

/-- Theorem stating the total weight of Courtney's marble collection --/
theorem courtney_marble_weight : 
  marbleCollectionWeight 80 (35/100) (45/100) (25/100) = 105 := by
  sorry

end NUMINAMATH_CALUDE_courtney_marble_weight_l2363_236364


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_73_l2363_236324

def p₁ (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1
def p₂ (x : ℝ) : ℝ := 2 * x^2 + x + 4
def p₃ (x : ℝ) : ℝ := x^2 + 2 * x + 3

def product (x : ℝ) : ℝ := p₁ x * p₂ x * p₃ x

theorem coefficient_x_cubed_is_73 :
  ∃ (a b c d : ℝ), product = fun x ↦ 73 * x^3 + a * x^4 + b * x^2 + c * x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_73_l2363_236324


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2363_236399

theorem rectangle_shorter_side 
  (width : Real) 
  (num_poles : Nat) 
  (pole_distance : Real) 
  (h1 : width = 50) 
  (h2 : num_poles = 24) 
  (h3 : pole_distance = 5) : 
  ∃ length : Real, 
    length = 7.5 ∧ 
    length ≤ width ∧ 
    2 * (length + width) = (num_poles - 1 : Real) * pole_distance := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2363_236399


namespace NUMINAMATH_CALUDE_initial_group_size_l2363_236367

/-- The number of men in the initial group -/
def initial_men_count : ℕ := sorry

/-- The average age increase when two women replace two men -/
def avg_age_increase : ℕ := 6

/-- The age of the first replaced man -/
def man1_age : ℕ := 18

/-- The age of the second replaced man -/
def man2_age : ℕ := 22

/-- The average age of the women -/
def women_avg_age : ℕ := 50

theorem initial_group_size : initial_men_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l2363_236367


namespace NUMINAMATH_CALUDE_positive_expressions_l2363_236366

theorem positive_expressions (U V W X Y : ℝ) 
  (h1 : U < V) (h2 : V < 0) (h3 : 0 < W) (h4 : W < X) (h5 : X < Y) : 
  (0 < U * V) ∧ 
  (0 < (X / V) * U) ∧ 
  (0 < W / (U * V)) ∧ 
  (0 < (X - Y) / W) := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l2363_236366


namespace NUMINAMATH_CALUDE_f_inequality_range_l2363_236303

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) - 1 / (x^2 + 2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2 * x - 1) ↔ 1/3 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2363_236303


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l2363_236354

/-- A quadratic function f(x) = x² + 2x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- The x-coordinate of the axis of symmetry for f -/
def axis_of_symmetry : ℝ := -1

theorem quadratic_points_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f c (-3) = y₁)
  (h₂ : f c (1/2) = y₂)
  (h₃ : f c 2 = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l2363_236354


namespace NUMINAMATH_CALUDE_employed_males_percentage_l2363_236327

theorem employed_males_percentage (total_population : ℝ) (employed_population : ℝ) (employed_females : ℝ) :
  employed_population = 0.64 * total_population →
  employed_females = 0.21875 * employed_population →
  0.4996 * total_population = employed_population - employed_females :=
by sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l2363_236327


namespace NUMINAMATH_CALUDE_area_equality_l2363_236395

-- Define a square
structure Square :=
  (A B C D : Point)

-- Define the property of being inside a square
def InsideSquare (P : Point) (s : Square) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality (s : Square) (P Q : Point) 
  (h_inside_P : InsideSquare P s)
  (h_inside_Q : InsideSquare Q s)
  (h_angle_PAQ : Angle s.A P Q = 45)
  (h_angle_PCQ : Angle s.C P Q = 45) :
  TriangleArea P s.A s.B + TriangleArea P s.C Q + TriangleArea Q s.A s.D =
  TriangleArea Q s.C s.D + TriangleArea P s.A Q + TriangleArea P s.B s.C :=
sorry

end NUMINAMATH_CALUDE_area_equality_l2363_236395


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2363_236389

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (b1 = 0 ∧ b2 = 0) ∨ (a1 * b2 = a2 * b1 ∧ a1 ≠ 0 ∧ a2 ≠ 0)

/-- The statement to be proved -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel (m - 2) (-1) 5 (m - 2) (3 - m) 2 → m = 2 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2363_236389


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l2363_236352

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l2363_236352


namespace NUMINAMATH_CALUDE_mistaken_divisor_l2363_236387

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 36 * correct_divisor →
  dividend = 63 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l2363_236387


namespace NUMINAMATH_CALUDE_range_of_f_l2363_236363

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 1 ≤ y ∧ y ≤ 10} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2363_236363


namespace NUMINAMATH_CALUDE_unique_n_modulo_101_l2363_236384

theorem unique_n_modulo_101 : ∃! n : ℤ, 0 ≤ n ∧ n < 101 ∧ (100 * n) % 101 = 72 % 101 ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_modulo_101_l2363_236384


namespace NUMINAMATH_CALUDE_train_distance_l2363_236322

/-- The distance covered by a train traveling at a constant speed for a given time. -/
theorem train_distance (speed : ℝ) (time : ℝ) (h1 : speed = 150) (h2 : time = 8) :
  speed * time = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2363_236322


namespace NUMINAMATH_CALUDE_cube_split_2015_l2363_236371

/-- The number of odd numbers in the "split" of n^3, for n ≥ 2 -/
def split_count (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number, starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_2015 (m : ℕ) (hm : m > 0) :
  (∃ k, k > 0 ∧ k ≤ split_count m ∧ nth_odd k = 2015) ↔ m = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_2015_l2363_236371


namespace NUMINAMATH_CALUDE_product_sum_difference_l2363_236360

theorem product_sum_difference (a b N : ℤ) : b = 7 → b - a = 2 → a * b = 2 * (a + b) + N → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_l2363_236360


namespace NUMINAMATH_CALUDE_store_pricing_strategy_l2363_236379

theorem store_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := 0.7 * list_price
  let marked_price := 1.07 * list_price
  let selling_price := 0.85 * marked_price
  selling_price = 1.3 * purchase_price :=
by sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l2363_236379


namespace NUMINAMATH_CALUDE_inequality_holds_l2363_236325

-- Define the real number a
variable (a : ℝ)

-- Define functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f, g, and a
axiom a_gt_one : a > 1
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_minus_g : ∀ x, f x - g x = a^x

-- State the theorem
theorem inequality_holds : g 0 < f 2 ∧ f 2 < f 3 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l2363_236325


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l2363_236332

/-- Given two concentric circles where the area between them is equal to twice the area of the smaller circle -/
theorem concentric_circles_theorem (a b : ℝ) (h : a > 0) (h' : b > 0) (h_concentric : a < b)
  (h_area : π * b^2 - π * a^2 = 2 * π * a^2) :
  (a / b = 1 / Real.sqrt 3) ∧ (π * a^2 / (π * b^2) = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l2363_236332


namespace NUMINAMATH_CALUDE_peach_apple_ratio_l2363_236342

/-- Given that Mr. Connell harvested 60 apples and the difference between
    the number of peaches and apples is 120, prove that the ratio of
    peaches to apples is 3:1. -/
theorem peach_apple_ratio :
  ∀ (peaches : ℕ),
  peaches - 60 = 120 →
  (peaches : ℚ) / 60 = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_peach_apple_ratio_l2363_236342


namespace NUMINAMATH_CALUDE_fixed_cost_satisfies_break_even_equation_l2363_236311

/-- The one-time fixed cost for a book publishing project -/
def fixed_cost : ℝ := 35678

/-- The variable cost per book -/
def variable_cost_per_book : ℝ := 11.50

/-- The selling price per book -/
def selling_price_per_book : ℝ := 20.25

/-- The number of books needed to break even -/
def break_even_quantity : ℕ := 4072

/-- Theorem stating that the fixed cost satisfies the break-even equation -/
theorem fixed_cost_satisfies_break_even_equation : 
  fixed_cost + (break_even_quantity : ℝ) * variable_cost_per_book = 
  (break_even_quantity : ℝ) * selling_price_per_book :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_satisfies_break_even_equation_l2363_236311


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2363_236368

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2363_236368


namespace NUMINAMATH_CALUDE_least_number_of_cans_l2363_236353

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 80)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let can_size := Nat.gcd maaza (Nat.gcd pepsi sprite)
  let total_cans := maaza / can_size + pepsi / can_size + sprite / can_size
  total_cans = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l2363_236353


namespace NUMINAMATH_CALUDE_final_ratio_is_two_to_one_l2363_236344

/-- Represents the ratio of milk to water in a mixture -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents a can containing a mixture of milk and water -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  mixture : Ratio

def add_milk (can : Can) (amount : ℕ) : Can :=
  { can with
    current_volume := can.current_volume + amount
    mixture := Ratio.mk (can.mixture.milk + amount) can.mixture.water
  }

theorem final_ratio_is_two_to_one
  (initial_can : Can)
  (h1 : initial_can.mixture = Ratio.mk 4 3)
  (h2 : initial_can.capacity = 36)
  (h3 : (add_milk initial_can 8).current_volume = initial_can.capacity) :
  (add_milk initial_can 8).mixture = Ratio.mk 2 1 := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_is_two_to_one_l2363_236344


namespace NUMINAMATH_CALUDE_a_4_value_l2363_236343

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 = 2 ∨ a 2 = 32) →
  (a 6 = 2 ∨ a 6 = 32) →
  a 2 * a 6 = 64 →
  a 2 + a 6 = 34 →
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_4_value_l2363_236343


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2363_236317

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 2 →                    -- a_1 = 2
  (a 1 + a 2 + a 3 = 26) →     -- S_3 = 26
  q = 3 ∨ q = -4 :=            -- conclusion: q is 3 or -4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2363_236317


namespace NUMINAMATH_CALUDE_faces_after_fifth_step_l2363_236357

/-- Represents the number of vertices at step n -/
def V : ℕ → ℕ
| 0 => 8
| n + 1 => 3 * V n

/-- Represents the number of faces at step n -/
def F : ℕ → ℕ
| 0 => 6
| n + 1 => F n + V n

/-- Theorem stating that the number of faces after the fifth step is 974 -/
theorem faces_after_fifth_step : F 5 = 974 := by
  sorry

end NUMINAMATH_CALUDE_faces_after_fifth_step_l2363_236357


namespace NUMINAMATH_CALUDE_cyclist_journey_time_l2363_236359

theorem cyclist_journey_time (a v : ℝ) (h1 : a > 0) (h2 : v > 0) (h3 : a / v = 5) :
  (a / (2 * v)) + (a / (2 * (1.25 * v))) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_time_l2363_236359


namespace NUMINAMATH_CALUDE_train_length_calculation_l2363_236334

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 240 → time_s = 21 → 
  ∃ (length_m : ℝ), abs (length_m - 1400.07) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2363_236334


namespace NUMINAMATH_CALUDE_trapezoidal_channel_bottom_width_l2363_236358

theorem trapezoidal_channel_bottom_width
  (top_width : ℝ)
  (area : ℝ)
  (depth : ℝ)
  (h_top_width : top_width = 12)
  (h_area : area = 700)
  (h_depth : depth = 70) :
  ∃ bottom_width : ℝ,
    bottom_width = 8 ∧
    area = (1 / 2) * (top_width + bottom_width) * depth :=
by sorry

end NUMINAMATH_CALUDE_trapezoidal_channel_bottom_width_l2363_236358


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2363_236308

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2*n - 2
  side2 : ℕ := 2*n
  side3 : ℕ := 2*n + 2

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.side1 + t.side2 + t.side3

/-- Triangle inequality for EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    (∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l2363_236308


namespace NUMINAMATH_CALUDE_empty_bucket_weight_l2363_236393

theorem empty_bucket_weight (M N P : ℝ) : ℝ :=
  let full_weight := M
  let three_quarters_weight := N
  let one_third_weight := P
  let empty_bucket_weight := 4 * N - 3 * M
  let water_weight := 4 * (M - N)
  
  have h1 : empty_bucket_weight + water_weight = full_weight := by sorry
  have h2 : empty_bucket_weight + 3/4 * water_weight = three_quarters_weight := by sorry
  have h3 : empty_bucket_weight + 1/3 * water_weight = one_third_weight := by sorry
  
  empty_bucket_weight

end NUMINAMATH_CALUDE_empty_bucket_weight_l2363_236393


namespace NUMINAMATH_CALUDE_equality_multiplication_negative_two_l2363_236335

theorem equality_multiplication_negative_two (m n : ℝ) : m = n → -2 * m = -2 * n := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_negative_two_l2363_236335


namespace NUMINAMATH_CALUDE_total_weight_equals_sum_l2363_236305

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := blue_ball_weight + brown_ball_weight

/-- Theorem: The total weight is equal to the sum of individual weights -/
theorem total_weight_equals_sum : total_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_equals_sum_l2363_236305


namespace NUMINAMATH_CALUDE_latest_start_time_l2363_236370

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Represents a turkey roasting scenario -/
structure TurkeyRoast where
  num_turkeys : ℕ
  turkey_weight : ℕ
  roast_time_per_pound : ℕ
  dinner_time : Time

def total_roast_time (tr : TurkeyRoast) : ℕ :=
  tr.num_turkeys * tr.turkey_weight * tr.roast_time_per_pound

def subtract_hours (t : Time) (h : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes - h * 60
  ⟨total_minutes / 60, total_minutes % 60, by sorry⟩

theorem latest_start_time (tr : TurkeyRoast) 
  (h_num : tr.num_turkeys = 2)
  (h_weight : tr.turkey_weight = 16)
  (h_roast_time : tr.roast_time_per_pound = 15)
  (h_dinner : tr.dinner_time = ⟨18, 0, by sorry⟩) :
  subtract_hours tr.dinner_time (total_roast_time tr / 60) = ⟨10, 0, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_latest_start_time_l2363_236370


namespace NUMINAMATH_CALUDE_exist_permutation_sum_all_nines_l2363_236365

/-- A function that checks if two natural numbers have the same digits (permutation) -/
def is_permutation (m n : ℕ) : Prop := sorry

/-- A function that checks if a natural number consists of all 9s -/
def all_nines (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of two natural numbers satisfying the given conditions -/
theorem exist_permutation_sum_all_nines : 
  ∃ (m n : ℕ), is_permutation m n ∧ all_nines (m + n) := by sorry

end NUMINAMATH_CALUDE_exist_permutation_sum_all_nines_l2363_236365


namespace NUMINAMATH_CALUDE_pentagon_star_area_theorem_l2363_236328

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The star formed by connecting every second vertex of the pentagon -/
def star (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set of points in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersect (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The quadrilateral APQD -/
def quadrilateral_APQD (p : RegularPentagon) : Set (ℝ × ℝ) :=
  let A := p.vertices 0
  let B := p.vertices 1
  let C := p.vertices 2
  let D := p.vertices 3
  let E := p.vertices 4
  let P := intersect A C B E
  let Q := intersect B D C E
  sorry

theorem pentagon_star_area_theorem (p : RegularPentagon) 
  (h : area (star p) = 1) : 
  area (quadrilateral_APQD p) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_star_area_theorem_l2363_236328


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l2363_236385

theorem no_function_satisfies_condition : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l2363_236385


namespace NUMINAMATH_CALUDE_sqrt_196_equals_14_l2363_236300

theorem sqrt_196_equals_14 : Real.sqrt 196 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_196_equals_14_l2363_236300


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l2363_236369

/-- Given a cuboid with dimensions a, b, and c, if its surface area is 11
    and the sum of the lengths of its twelve edges is 24,
    then the length of its diagonal is 5. -/
theorem cuboid_diagonal (a b c : ℝ) 
    (h1 : 2 * (a * b + b * c + a * c) = 11)  -- surface area condition
    (h2 : 4 * (a + b + c) = 24) :            -- sum of edges condition
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l2363_236369


namespace NUMINAMATH_CALUDE_elective_schemes_count_l2363_236362

/-- The number of courses offered -/
def total_courses : ℕ := 9

/-- The number of mutually exclusive courses -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of different elective schemes -/
def elective_schemes : ℕ := 75

theorem elective_schemes_count :
  (Nat.choose exclusive_courses 1 * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  (Nat.choose (total_courses - exclusive_courses) courses_to_choose) = elective_schemes :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l2363_236362


namespace NUMINAMATH_CALUDE_number_divisible_by_six_l2363_236376

theorem number_divisible_by_six : ∃ n : ℕ, n % 6 = 0 ∧ n / 6 = 209 → n = 1254 := by
  sorry

end NUMINAMATH_CALUDE_number_divisible_by_six_l2363_236376


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l2363_236382

theorem polynomial_difference_divisibility
  (p : Polynomial ℤ) (b c : ℤ) (h : b ≠ c) :
  (b - c) ∣ (p.eval b - p.eval c) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l2363_236382


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2363_236329

/-- The total surface area of a hemisphere with radius 9 cm, including its circular base, is 243π cm². -/
theorem hemisphere_surface_area :
  let r : ℝ := 9
  let base_area : ℝ := π * r^2
  let curved_area : ℝ := 2 * π * r^2
  let total_area : ℝ := base_area + curved_area
  total_area = 243 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2363_236329


namespace NUMINAMATH_CALUDE_subtraction_reciprocal_l2363_236347

theorem subtraction_reciprocal (x y : ℝ) (h : x - y = 3 * x * y) :
  1 / x - 1 / y = -3 :=
by sorry

end NUMINAMATH_CALUDE_subtraction_reciprocal_l2363_236347


namespace NUMINAMATH_CALUDE_last_digit_of_total_edge_count_l2363_236383

/-- Represents an 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents a 1x2 domino piece -/
def Domino := Σ' (i : Fin 8) (j : Fin 7), Unit

/-- A tiling of the chessboard with dominos -/
def Tiling := Chessboard → Option Domino

/-- The number of valid tilings of the chessboard -/
def numTilings : ℕ := 12988816

/-- An edge of the chessboard -/
inductive Edge
| horizontal (i : Fin 9) (j : Fin 8) : Edge
| vertical (i : Fin 8) (j : Fin 9) : Edge

/-- The number of tilings that include a given edge -/
def edgeCount (e : Edge) : ℕ := sorry

/-- The sum of edgeCount for all edges -/
def totalEdgeCount : ℕ := sorry

/-- Theorem: The last digit of totalEdgeCount is 4 -/
theorem last_digit_of_total_edge_count :
  totalEdgeCount % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_total_edge_count_l2363_236383


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l2363_236337

/-- The length of the tangent from the origin to a circle passing through specific points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (4, 5) → B = (8, 10) → C = (7, 17) → 
  ∃ (circle : Set (ℝ × ℝ)) (tangent : ℝ × ℝ → ℝ),
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (tangent (0, 0) = 2 * Real.sqrt 41) := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l2363_236337


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2363_236341

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + 4 ≥ D * (x + y + z)) ↔ D ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2363_236341


namespace NUMINAMATH_CALUDE_min_value_x_plus_half_y_l2363_236355

theorem min_value_x_plus_half_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - 2 * x - y = 0) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b - 2 * a - b = 0 → x + y / 2 ≤ a + b / 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x + y / 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_half_y_l2363_236355


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2363_236345

theorem fraction_sum_equality : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) = (348 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2363_236345


namespace NUMINAMATH_CALUDE_triangle_inequality_l2363_236381

/-- Given a triangle with side lengths a, b, and c, 
    prove the inequality and its equality condition --/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0) ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2363_236381


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_plus_x_l2363_236330

/-- A line y = mx + b is tangent to a curve y = f(x) at point (x₀, f(x₀)) if:
    1. The line passes through the point (x₀, f(x₀))
    2. The slope of the line equals the derivative of f at x₀ -/
def is_tangent_line (f : ℝ → ℝ) (f' : ℝ → ℝ) (m b x₀ : ℝ) : Prop :=
  f x₀ = m * x₀ + b ∧ f' x₀ = m

theorem tangent_line_to_exp_plus_x (b : ℝ) :
  (∃ x₀ : ℝ, is_tangent_line (λ x => Real.exp x + x) (λ x => Real.exp x + 1) 2 b x₀) →
  b = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_plus_x_l2363_236330


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l2363_236336

/-- Given three points A, B, and C in a plane satisfying certain conditions,
    prove that the sum of coordinates of A is 22. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 8) →
  C = (5, 11) →
  A.1 + A.2 = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l2363_236336


namespace NUMINAMATH_CALUDE_rainwater_chickens_l2363_236315

/-- Proves that Mr. Rainwater has 18 chickens given the conditions -/
theorem rainwater_chickens :
  ∀ (goats cows chickens : ℕ),
    cows = 9 →
    goats = 4 * cows →
    goats = 2 * chickens →
    chickens = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainwater_chickens_l2363_236315


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l2363_236388

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l2363_236388


namespace NUMINAMATH_CALUDE_probability_of_specific_dice_outcome_l2363_236350

def num_dice : ℕ := 5
def num_sides : ℕ := 5
def target_number : ℕ := 3
def num_target : ℕ := 2

theorem probability_of_specific_dice_outcome :
  (num_dice.choose num_target *
   (1 / num_sides) ^ num_target *
   ((num_sides - 1) / num_sides) ^ (num_dice - num_target) : ℚ) =
  640 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_dice_outcome_l2363_236350


namespace NUMINAMATH_CALUDE_tangent_to_exponential_l2363_236340

theorem tangent_to_exponential (k : ℝ) :
  (∃ x : ℝ, k * x = Real.exp x ∧ k = Real.exp x) → k = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_to_exponential_l2363_236340


namespace NUMINAMATH_CALUDE_right_triangle_area_l2363_236397

/-- The area of a right triangle given the sum of its legs and the altitude from the right angle. -/
theorem right_triangle_area (l h : ℝ) (hl : l > 0) (hh : h > 0) :
  ∃ S : ℝ, S = (1/2) * h * (Real.sqrt (l^2 + h^2) - h) ∧ 
  S > 0 ∧ 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = l ∧ 
  S = (1/2) * x * h ∧ S = (1/2) * y * h :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_area_l2363_236397


namespace NUMINAMATH_CALUDE_quadratic_roots_l2363_236377

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2363_236377


namespace NUMINAMATH_CALUDE_set_intersection_proof_l2363_236309

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem set_intersection_proof : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l2363_236309


namespace NUMINAMATH_CALUDE_fish_in_pond_l2363_236318

theorem fish_in_pond (tagged_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  tagged_fish = 60 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = tagged_fish / (1500 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_fish_in_pond_l2363_236318


namespace NUMINAMATH_CALUDE_abcd_equation_solutions_l2363_236301

theorem abcd_equation_solutions :
  ∀ (A B C D : ℕ),
    0 ≤ A ∧ A ≤ 9 ∧
    0 ≤ B ∧ B ≤ 9 ∧
    0 ≤ C ∧ C ≤ 9 ∧
    0 ≤ D ∧ D ≤ 9 ∧
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧
    1000 * A + 100 * B + 10 * C + D = (10 * A + D) * (101 * A + 10 * D) →
    (A = 1 ∧ B = 0 ∧ C = 1 ∧ D = 0) ∨
    (A = 1 ∧ B = 2 ∧ C = 2 ∧ D = 1) ∨
    (A = 1 ∧ B = 4 ∧ C = 5 ∧ D = 2) ∨
    (A = 1 ∧ B = 7 ∧ C = 0 ∧ D = 3) ∨
    (A = 1 ∧ B = 9 ∧ C = 7 ∧ D = 4) :=
by sorry

end NUMINAMATH_CALUDE_abcd_equation_solutions_l2363_236301


namespace NUMINAMATH_CALUDE_inverse_34_mod_47_l2363_236391

theorem inverse_34_mod_47 (h : (13⁻¹ : ZMod 47) = 29) : (34⁻¹ : ZMod 47) = 18 := by
  sorry

end NUMINAMATH_CALUDE_inverse_34_mod_47_l2363_236391


namespace NUMINAMATH_CALUDE_box_surface_area_l2363_236378

def sheet_length : ℕ := 25
def sheet_width : ℕ := 40
def corner_size : ℕ := 8

def surface_area : ℕ :=
  sheet_length * sheet_width - 4 * (corner_size * corner_size)

theorem box_surface_area :
  surface_area = 744 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l2363_236378


namespace NUMINAMATH_CALUDE_birds_in_tree_l2363_236398

theorem birds_in_tree (initial_birds new_birds : ℕ) 
  (h1 : initial_birds = 14) 
  (h2 : new_birds = 21) : 
  initial_birds + new_birds = 35 :=
by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2363_236398


namespace NUMINAMATH_CALUDE_hexagon_triangle_perimeter_ratio_l2363_236304

theorem hexagon_triangle_perimeter_ratio :
  ∀ (s_h s_t : ℝ),
  s_h > 0 → s_t > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * ((3 * s_h^2 * Real.sqrt 3) / 2) →
  (3 * s_t) / (6 * s_h) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hexagon_triangle_perimeter_ratio_l2363_236304


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l2363_236314

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 200 × 325 × 376 rectangular solid, an internal diagonal passes through 868 unit cubes -/
theorem internal_diagonal_cubes : cubes_passed 200 325 376 = 868 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l2363_236314


namespace NUMINAMATH_CALUDE_watch_correction_l2363_236310

/-- Represents the time difference between two dates in hours -/
def timeDifference (startDate endDate : Nat) : Nat :=
  (endDate - startDate) * 24

/-- Represents the additional hours on the last day -/
def additionalHours (startHour endHour : Nat) : Nat :=
  endHour - startHour

/-- Calculates the total hours elapsed -/
def totalHours (daysDifference additionalHours : Nat) : Nat :=
  daysDifference + additionalHours

/-- Converts daily time loss to hourly time loss -/
def hourlyLoss (dailyLoss : Rat) : Rat :=
  dailyLoss / 24

/-- Calculates the total time loss -/
def totalLoss (hourlyLoss : Rat) (totalHours : Nat) : Rat :=
  hourlyLoss * totalHours

theorem watch_correction (watchLoss : Rat) (startDate endDate startHour endHour : Nat) :
  watchLoss = 3.75 →
  startDate = 15 →
  endDate = 24 →
  startHour = 10 →
  endHour = 16 →
  totalLoss (hourlyLoss watchLoss) (totalHours (timeDifference startDate endDate) (additionalHours startHour endHour)) = 34.6875 := by
  sorry

#check watch_correction

end NUMINAMATH_CALUDE_watch_correction_l2363_236310


namespace NUMINAMATH_CALUDE_container_height_l2363_236306

/-- The height of a cylindrical container A, given specific conditions --/
theorem container_height (r_A r_B : ℝ) (h : ℝ → ℝ) :
  r_A = 2 →
  r_B = 3 →
  (∀ x, h x = (2/3 * x - 6)) →
  (π * r_A^2 * x = π * r_B^2 * h x) →
  x = 27 :=
by sorry

end NUMINAMATH_CALUDE_container_height_l2363_236306


namespace NUMINAMATH_CALUDE_inequality_proof_l2363_236361

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2363_236361


namespace NUMINAMATH_CALUDE_min_value_implies_a_equals_two_l2363_236302

theorem min_value_implies_a_equals_two (x y a : ℝ) :
  x + 3*y + 5 ≥ 0 →
  x + y - 1 ≤ 0 →
  x + a ≥ 0 →
  (∀ x' y', x' + 3*y' + 5 ≥ 0 → x' + y' - 1 ≤ 0 → x' + 2*y' ≥ x + 2*y) →
  x + 2*y = -4 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_equals_two_l2363_236302


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l2363_236321

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 : ℝ) (m2 n2 : ℝ) : Prop := m1 * n2 = m2 * n1

/-- The line ax+3y+1=0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- The line 2x+(a+1)y+1=0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- The main theorem: the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three (a : ℝ) : 
  parallel a 3 2 (a + 1) ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l2363_236321


namespace NUMINAMATH_CALUDE_area_between_sine_and_constant_line_l2363_236349

theorem area_between_sine_and_constant_line : 
  let f : ℝ → ℝ := λ x => Real.sin x
  let g : ℝ → ℝ := λ _ => (1/2 : ℝ)
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := Real.pi
  ∃ (area : ℝ), area = ∫ x in lower_bound..upper_bound, |f x - g x| ∧ area = Real.sqrt 3 - Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_sine_and_constant_line_l2363_236349


namespace NUMINAMATH_CALUDE_phenol_red_identifies_urea_decomposing_bacteria_l2363_236356

/-- Represents different types of reagents --/
inductive Reagent
  | PhenolRed
  | EMB
  | SudanIII
  | Biuret

/-- Represents a culture medium --/
structure CultureMedium where
  nitrogenSource : String
  reagent : Reagent

/-- Represents the result of a bacterial identification test --/
inductive TestResult
  | Positive
  | Negative

/-- Function to perform urea decomposition test --/
def ureaDecompositionTest (medium : CultureMedium) : TestResult := sorry

/-- Theorem stating that phenol red is the correct reagent for identifying urea-decomposing bacteria --/
theorem phenol_red_identifies_urea_decomposing_bacteria :
  ∀ (medium : CultureMedium),
    medium.nitrogenSource = "urea" →
    medium.reagent = Reagent.PhenolRed →
    ureaDecompositionTest medium = TestResult.Positive :=
  sorry

end NUMINAMATH_CALUDE_phenol_red_identifies_urea_decomposing_bacteria_l2363_236356
