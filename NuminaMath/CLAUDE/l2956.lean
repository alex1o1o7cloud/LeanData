import Mathlib

namespace megatech_budget_allocation_l2956_295693

theorem megatech_budget_allocation :
  let total_budget : ℝ := 100
  let microphotonics : ℝ := 10
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let industrial_lubricants : ℝ := 8
  let basic_astrophysics_degrees : ℝ := 50.4
  let total_degrees : ℝ := 360
  let basic_astrophysics : ℝ := (basic_astrophysics_degrees / total_degrees) * total_budget
  let genetically_modified_microorganisms : ℝ := total_budget - (microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics)
  genetically_modified_microorganisms = 29 := by
sorry

end megatech_budget_allocation_l2956_295693


namespace percent_composition_l2956_295690

theorem percent_composition (z : ℝ) (hz : z ≠ 0) :
  (42 / 100) * z = (60 / 100) * ((70 / 100) * z) := by
  sorry

end percent_composition_l2956_295690


namespace lucy_shells_count_l2956_295692

/-- The number of shells Lucy initially had -/
def initial_shells : ℕ := 68

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := initial_shells + additional_shells

theorem lucy_shells_count : total_shells = 89 := by
  sorry

end lucy_shells_count_l2956_295692


namespace half_shading_sufficient_l2956_295670

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (total_cells : ℕ)

/-- Represents the minimum number of cells to be shaded --/
def min_shaded_cells (g : Grid) : ℕ := g.total_cells / 2

/-- Theorem stating that shading half the cells is sufficient --/
theorem half_shading_sufficient (g : Grid) (h : g.size = 12) (h' : g.total_cells = 144) :
  ∃ (shaded : ℕ), shaded = min_shaded_cells g ∧ 
  shaded ≤ g.total_cells ∧
  shaded ≥ g.total_cells / 2 :=
sorry

#check half_shading_sufficient

end half_shading_sufficient_l2956_295670


namespace ac_unit_final_price_l2956_295635

/-- Calculates the final price of an air-conditioning unit after a series of price changes. -/
def final_price (initial_price : ℝ) : ℝ :=
  let price1 := initial_price * (1 - 0.12)  -- February
  let price2 := price1 * (1 + 0.08)         -- March
  let price3 := price2 * (1 - 0.10)         -- April
  let price4 := price3 * (1 + 0.05)         -- June
  let price5 := price4 * (1 - 0.07)         -- August
  let price6 := price5 * (1 + 0.06)         -- October
  let price7 := price6 * (1 - 0.15)         -- November
  price7

/-- Theorem stating that the final price of the air-conditioning unit is approximately $353.71. -/
theorem ac_unit_final_price : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |final_price 470 - 353.71| < ε :=
sorry

end ac_unit_final_price_l2956_295635


namespace cone_radius_l2956_295664

/-- Given a cone with slant height 10 cm and curved surface area 157.07963267948966 cm², 
    the radius of the base is 5 cm. -/
theorem cone_radius (slant_height : ℝ) (curved_surface_area : ℝ) :
  slant_height = 10 ∧ 
  curved_surface_area = 157.07963267948966 ∧
  curved_surface_area = Real.pi * (5 : ℝ) * slant_height :=
by sorry

end cone_radius_l2956_295664


namespace roxanne_change_l2956_295676

/-- Calculates the change Roxanne should receive after buying lemonade and sandwiches -/
theorem roxanne_change (lemonade_price : ℝ) (sandwich_price : ℝ) (lemonade_quantity : ℕ) (sandwich_quantity : ℕ) (paid_amount : ℝ) : 
  lemonade_price = 2 →
  sandwich_price = 2.5 →
  lemonade_quantity = 2 →
  sandwich_quantity = 2 →
  paid_amount = 20 →
  paid_amount - (lemonade_price * lemonade_quantity + sandwich_price * sandwich_quantity) = 11 := by
sorry

end roxanne_change_l2956_295676


namespace triangle_with_integer_altitudes_and_prime_inradius_l2956_295632

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Calculates the semi-perimeter of a triangle -/
def semiPerimeter (t : Triangle) : ℚ :=
  (t.a.val + t.b.val + t.c.val) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : Triangle) : ℚ :=
  let s := semiPerimeter t
  (s * (s - t.a.val) * (s - t.b.val) * (s - t.c.val)).sqrt

/-- Calculates the inradius of a triangle -/
def inradius (t : Triangle) : ℚ :=
  area t / semiPerimeter t

/-- Calculates the altitude to side a of a triangle -/
def altitudeA (t : Triangle) : ℚ :=
  2 * area t / t.a.val

/-- Calculates the altitude to side b of a triangle -/
def altitudeB (t : Triangle) : ℚ :=
  2 * area t / t.b.val

/-- Calculates the altitude to side c of a triangle -/
def altitudeC (t : Triangle) : ℚ :=
  2 * area t / t.c.val

/-- States that a number is prime -/
def isPrime (n : ℕ) : Prop :=
  Nat.Prime n

theorem triangle_with_integer_altitudes_and_prime_inradius :
  ∃ (t : Triangle),
    t.a = 13 ∧ t.b = 14 ∧ t.c = 15 ∧
    (altitudeA t).isInt ∧ (altitudeB t).isInt ∧ (altitudeC t).isInt ∧
    ∃ (r : ℕ), (inradius t) = r ∧ isPrime r :=
by sorry

end triangle_with_integer_altitudes_and_prime_inradius_l2956_295632


namespace quadratic_function_properties_l2956_295631

/-- A quadratic function that opens upwards and passes through (0,1) -/
def QuadraticFunction (a b : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

theorem quadratic_function_properties (a b : ℝ) (h : a > 0) :
  (QuadraticFunction a b h) 0 = 1 ∧
  ∀ x y : ℝ, x < y → (QuadraticFunction a b h) x < (QuadraticFunction a b h) y :=
by sorry

end quadratic_function_properties_l2956_295631


namespace twentyFirstTerm_l2956_295607

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 21st term of an arithmetic progression with first term 3 and common difference 5 is 103 -/
theorem twentyFirstTerm :
  arithmeticProgressionTerm 3 5 21 = 103 := by
  sorry

end twentyFirstTerm_l2956_295607


namespace smallest_integer_inequality_l2956_295683

theorem smallest_integer_inequality (x y z : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c : ℝ), (a^2 + b^2 + c^2)^2 > m * (a^4 + b^4 + c^4) :=
by
  sorry

end smallest_integer_inequality_l2956_295683


namespace max_value_of_sum_cube_roots_l2956_295662

theorem max_value_of_sum_cube_roots (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_100 : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + 
           (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7 ^ (1/3) := by
sorry

end max_value_of_sum_cube_roots_l2956_295662


namespace f_at_2_l2956_295623

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem f_at_2 : f 2 = 15 := by sorry

end f_at_2_l2956_295623


namespace maize_donated_amount_l2956_295608

/-- The amount of maize donated to Alfred -/
def maize_donated (
  stored_per_month : ℕ)  -- Amount of maize stored per month
  (months : ℕ)           -- Number of months
  (stolen : ℕ)           -- Amount of maize stolen
  (final_amount : ℕ)     -- Final amount of maize after 2 years
  : ℕ :=
  final_amount - (stored_per_month * months - stolen)

/-- Theorem stating the amount of maize donated to Alfred -/
theorem maize_donated_amount :
  maize_donated 1 24 5 27 = 8 := by
  sorry

end maize_donated_amount_l2956_295608


namespace sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l2956_295629

theorem sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l2956_295629


namespace integer_between_sqrt2_and_sqrt12_l2956_295627

theorem integer_between_sqrt2_and_sqrt12 (a : ℤ) : 
  (Real.sqrt 2 < a) ∧ (a < Real.sqrt 12) → (a = 2 ∨ a = 3) := by
  sorry

end integer_between_sqrt2_and_sqrt12_l2956_295627


namespace garage_sale_items_l2956_295618

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card > 0 → 
  radio_price ∈ prices → 
  (prices.filter (λ p => p > radio_price)).card = 16 → 
  (prices.filter (λ p => p < radio_price)).card = 23 → 
  prices.card = 40 := by
sorry

end garage_sale_items_l2956_295618


namespace conical_frustum_volume_l2956_295667

/-- Right prism with equilateral triangle base -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Conical frustum within the right prism -/
def ConicalFrustum (p : RightPrism) : Type :=
  Unit

/-- Volume of the conical frustum -/
def volume (p : RightPrism) (f : ConicalFrustum p) : ℝ :=
  sorry

/-- Theorem: Volume of conical frustum in given right prism -/
theorem conical_frustum_volume (p : RightPrism) (f : ConicalFrustum p)
    (h1 : p.height = 3)
    (h2 : p.base_side = 1) :
    volume p f = Real.sqrt 3 / 4 := by
  sorry

end conical_frustum_volume_l2956_295667


namespace sqrt_product_equality_l2956_295651

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2956_295651


namespace root_magnitude_theorem_l2956_295679

theorem root_magnitude_theorem (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + (A+C)/2*x + (B+D)/2 = 0 → Complex.abs x < 1 :=
sorry

end root_magnitude_theorem_l2956_295679


namespace inscribed_rectangle_perimeter_is_12_l2956_295654

/-- A right triangle with legs of length 6 containing an inscribed rectangle -/
structure RightTriangleWithInscribedRectangle where
  /-- The length of each leg of the right triangle -/
  leg_length : ℝ
  /-- The inscribed rectangle shares an angle with the triangle -/
  shares_angle : Bool
  /-- The inscribed rectangle is contained within the triangle -/
  is_inscribed : Bool

/-- The perimeter of the inscribed rectangle -/
def inscribed_rectangle_perimeter (t : RightTriangleWithInscribedRectangle) : ℝ := 12

/-- Theorem: The perimeter of the inscribed rectangle is 12 -/
theorem inscribed_rectangle_perimeter_is_12 (t : RightTriangleWithInscribedRectangle)
  (h1 : t.leg_length = 6)
  (h2 : t.shares_angle = true)
  (h3 : t.is_inscribed = true) :
  inscribed_rectangle_perimeter t = 12 := by sorry

end inscribed_rectangle_perimeter_is_12_l2956_295654


namespace daps_equiv_48_dips_l2956_295656

/-- Conversion rate between daps and dops -/
def daps_to_dops : ℚ := 4 / 5

/-- Conversion rate between dops and dips -/
def dops_to_dips : ℚ := 8 / 3

/-- The number of daps equivalent to 48 dips -/
def daps_equiv_to_48_dips : ℚ := 22.5

theorem daps_equiv_48_dips :
  daps_equiv_to_48_dips = 48 * dops_to_dips * daps_to_dops := by
  sorry

end daps_equiv_48_dips_l2956_295656


namespace odd_sum_selections_count_l2956_295603

/-- The number of ways to select k elements from n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 11 -/
def ballNumbers : Finset ℕ := sorry

/-- The number of odd numbers in ballNumbers -/
def oddCount : ℕ := sorry

/-- The number of even numbers in ballNumbers -/
def evenCount : ℕ := sorry

/-- The number of ways to select 5 balls with an odd sum -/
def oddSumSelections : ℕ := sorry

theorem odd_sum_selections_count :
  oddSumSelections = 236 := by sorry

end odd_sum_selections_count_l2956_295603


namespace cube_root_of_negative_one_l2956_295621

theorem cube_root_of_negative_one : ∃ x : ℝ, x^3 = -1 ∧ x = -1 := by sorry

end cube_root_of_negative_one_l2956_295621


namespace perpendicular_line_through_B_l2956_295624

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point B
def point_B : ℝ × ℝ := (3, 0)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Theorem statement
theorem perpendicular_line_through_B :
  (perpendicular_line point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (x - point_B.1) * (x - point_B.1) + (y - point_B.2) * (y - point_B.2) ≠ 0) :=
sorry

end perpendicular_line_through_B_l2956_295624


namespace locus_of_T_l2956_295611

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a rectangle
structure Rectangle where
  m : Point
  k : Point
  t : Point
  p : Point

-- Main theorem
theorem locus_of_T (c : Circle) (m : Point) 
  (h1 : (m.x - c.center.1)^2 + (m.y - c.center.2)^2 < c.radius^2) :
  ∃ (c_locus : Circle),
    c_locus.center = c.center ∧
    c_locus.radius = Real.sqrt (2 * c.radius^2 - (m.x^2 + m.y^2)) ∧
    ∀ (rect : Rectangle),
      (rect.m = m) →
      ((rect.k.x - c.center.1)^2 + (rect.k.y - c.center.2)^2 = c.radius^2) →
      ((rect.p.x - c.center.1)^2 + (rect.p.y - c.center.2)^2 = c.radius^2) →
      (rect.m.x - rect.t.x = rect.k.x - rect.p.x) →
      (rect.m.y - rect.t.y = rect.k.y - rect.p.y) →
      ((rect.t.x - c.center.1)^2 + (rect.t.y - c.center.2)^2 = c_locus.radius^2) :=
by
  sorry


end locus_of_T_l2956_295611


namespace units_digit_of_sum_factorials_l2956_295642

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorials :
  units_digit (sum_of_factorials 50) = 3 := by sorry

end units_digit_of_sum_factorials_l2956_295642


namespace f_negative_2011_l2956_295671

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

theorem f_negative_2011 (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end f_negative_2011_l2956_295671


namespace exam_arrangements_l2956_295666

/- Define the number of subjects -/
def num_subjects : ℕ := 6

/- Define the condition that Chinese must be first -/
def chinese_first : ℕ := 1

/- Define the number of subjects excluding Chinese, Math, and English -/
def other_subjects : ℕ := 3

/- Define the number of spaces available for Math and English -/
def available_spaces : ℕ := 4

/- Define the function to calculate the number of arrangements -/
def num_arrangements : ℕ :=
  chinese_first * (Nat.factorial other_subjects) * (available_spaces * (available_spaces - 1) / 2)

/- Theorem statement -/
theorem exam_arrangements :
  num_arrangements = 72 :=
sorry

end exam_arrangements_l2956_295666


namespace quadratic_real_roots_range_l2956_295696

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end quadratic_real_roots_range_l2956_295696


namespace a_zero_sufficient_a_zero_not_necessary_l2956_295604

def f (a b x : ℝ) : ℝ := x^2 + a * abs x + b

-- Sufficient condition
theorem a_zero_sufficient (a b : ℝ) :
  a = 0 → ∀ x, f a b x = f a b (-x) :=
sorry

-- Not necessary condition
theorem a_zero_not_necessary :
  ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f a b x = f a b (-x)) :=
sorry

end a_zero_sufficient_a_zero_not_necessary_l2956_295604


namespace cindys_calculation_l2956_295625

theorem cindys_calculation (x : ℝ) : (x - 7) / 5 = 57 → (x - 5) / 7 = 41 := by
  sorry

end cindys_calculation_l2956_295625


namespace waiter_customer_count_l2956_295617

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 → lunch_rush = 20.0 → later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end waiter_customer_count_l2956_295617


namespace f_increasing_f_comparison_l2956_295691

noncomputable section

-- Define the function f with the given property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → a * f a + b * f b > a * f b + b * f a

-- Theorem 1: f is monotonically increasing
theorem f_increasing (f : ℝ → ℝ) (hf : f_property f) :
  Monotone f := by sorry

-- Theorem 2: f(x+y) > f(6) under given conditions
theorem f_comparison (f : ℝ → ℝ) (hf : f_property f) (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_eq : 4/x + 9/y = 4) :
  f (x + y) > f 6 := by sorry

end f_increasing_f_comparison_l2956_295691


namespace quadratic_factorization_l2956_295684

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2) (x - 2)

/-- The equation m^2 - 4 = (m + 2)(m - 2) represents factorization from left to right -/
theorem quadratic_factorization :
  is_factorization_left_to_right (λ m => m^2 - 4) (λ a b => a * b) :=
sorry

end quadratic_factorization_l2956_295684


namespace min_value_expression_l2956_295614

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min :=
by sorry

end min_value_expression_l2956_295614


namespace diophantine_equation_solutions_l2956_295661

def solutions : Set (ℤ × ℤ) := {(6, 9), (7, 3), (8, 1), (9, 0), (11, -1), (17, -2), (4, -15), (3, -9), (2, -7), (1, -6), (-1, -5), (-7, -4)}

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | x * y + 3 * x - 5 * y = -3} = solutions := by sorry

end diophantine_equation_solutions_l2956_295661


namespace smallest_common_factor_l2956_295605

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 7 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*7 - 3) ∧ k ∣ (5*7 + 4)) :=
by sorry

end smallest_common_factor_l2956_295605


namespace CD_length_l2956_295665

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def AD_perp_AB : (A.1 - D.1) * (B.1 - A.1) + (A.2 - D.2) * (B.2 - A.2) = 0 := sorry
def BC_perp_AB : (B.1 - C.1) * (B.1 - A.1) + (B.2 - C.2) * (B.2 - A.2) = 0 := sorry
def CD_perp_AC : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0 := sorry

def AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 := sorry
def BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3 := sorry

-- Theorem to prove
theorem CD_length : 
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 20/3 :=
by sorry

end CD_length_l2956_295665


namespace excellent_grade_percentage_l2956_295678

theorem excellent_grade_percentage (total : ℕ) (excellent : ℕ) (h1 : total = 360) (h2 : excellent = 72) :
  (excellent : ℚ) / (total : ℚ) * 100 = 20 := by
  sorry

end excellent_grade_percentage_l2956_295678


namespace intersection_and_union_of_A_and_B_l2956_295619

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A with respect to U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B with respect to U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem intersection_and_union_of_A_and_B :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end intersection_and_union_of_A_and_B_l2956_295619


namespace multiple_births_quintuplets_l2956_295633

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 3 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_quintuplets_quadruplets : ∃ (q qu : ℕ), q = qu / 2)
  (h_sum : ∃ (w t q qu : ℕ), 2 * w + 3 * t + 4 * q + 5 * qu = total_babies) :
  ∃ (quintuplets : ℕ), quintuplets = 1500 / 11 ∧ 
    quintuplets * 5 = total_babies * 5 / 11 :=
by sorry

end multiple_births_quintuplets_l2956_295633


namespace factor_expression_l2956_295668

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factor_expression_l2956_295668


namespace ant_final_position_l2956_295648

/-- Represents the vertices of the rectangle --/
inductive Vertex : Type
  | A : Vertex
  | B : Vertex
  | C : Vertex
  | D : Vertex

/-- Represents a single movement of the ant --/
def next_vertex : Vertex → Vertex
  | Vertex.A => Vertex.B
  | Vertex.B => Vertex.C
  | Vertex.C => Vertex.D
  | Vertex.D => Vertex.A

/-- Represents multiple movements of the ant --/
def ant_position (start : Vertex) (moves : Nat) : Vertex :=
  match moves with
  | 0 => start
  | n + 1 => next_vertex (ant_position start n)

/-- The main theorem to prove --/
theorem ant_final_position :
  ant_position Vertex.A 2018 = Vertex.C := by
  sorry

end ant_final_position_l2956_295648


namespace inequality_solution_existence_l2956_295610

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Icc (-9/4 : ℝ) 2 := by sorry

end inequality_solution_existence_l2956_295610


namespace min_garden_cost_l2956_295672

/-- Represents a rectangular region in the flower bed -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower -/
structure Flower where
  name : String
  price : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of filling a region with a specific flower -/
def cost (r : Region) (f : Flower) : ℝ := area r * f.price

/-- The flower bed arrangement -/
def flowerBed : List Region := [
  { length := 5, width := 2 },
  { length := 4, width := 2 },
  { length := 7, width := 4 },
  { length := 3, width := 5 }
]

/-- Available flower types -/
def flowers : List Flower := [
  { name := "Fuchsia", price := 3.5 },
  { name := "Gardenia", price := 4 },
  { name := "Canna", price := 2 },
  { name := "Begonia", price := 1.5 }
]

/-- Theorem stating the minimum cost of the garden -/
theorem min_garden_cost :
  ∃ (arrangement : List (Region × Flower)),
    arrangement.length = flowerBed.length ∧
    (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ arrangement) ∧
    (arrangement.map (λ (r, f) => cost r f)).sum = 140 ∧
    ∀ (other_arrangement : List (Region × Flower)),
      other_arrangement.length = flowerBed.length →
      (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ other_arrangement) →
      (other_arrangement.map (λ (r, f) => cost r f)).sum ≥ 140 := by
  sorry

end min_garden_cost_l2956_295672


namespace polynomial_integer_roots_l2956_295644

def polynomial (x a : ℤ) : ℤ := x^3 + 5*x^2 + a*x + 12

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x a = 0

def valid_a_values : Set ℤ := {-18, 16, -20, 12, -16, 8, -11, 5, -4, 2, 0, -1}

theorem polynomial_integer_roots :
  ∀ a : ℤ, has_integer_root a ↔ a ∈ valid_a_values := by sorry

end polynomial_integer_roots_l2956_295644


namespace hyperbola_equation_l2956_295663

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    prove that under certain conditions, its equation is x²/4 - y²/6 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (C : ℝ × ℝ → Prop) 
  (hC : ∀ x y, C (x, y) ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ) (hF : F.1 > 0 ∧ F.2 = 0) -- Right focus
  (B : ℝ × ℝ) (hB : B.1 = 0) -- B is on the imaginary axis
  (A : ℝ × ℝ) (hA : C A) -- A is on the hyperbola
  (hAF : ∃ t : ℝ, A = B + t • (F - B)) -- A is on BF
  (hBA : ∃ k : ℝ, k • (A - B) = 2 • (F - A)) -- BA = 2AF
  (hBF : (F.1 - B.1)^2 + (F.2 - B.2)^2 = 16) -- |BF| = 4
  : ∀ x y, C (x, y) ↔ x^2 / 4 - y^2 / 6 = 1 :=
by sorry

end hyperbola_equation_l2956_295663


namespace sum_of_repeating_decimals_l2956_295634

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_04 : ℚ := 4/99
def repeating_decimal_005 : ℚ := 5/999

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_04 + repeating_decimal_005 = 742/999 := by
  sorry

end sum_of_repeating_decimals_l2956_295634


namespace g_neg_one_value_l2956_295613

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function when combined with x^2
def isOddWithSquare (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : isOddWithSquare f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end g_neg_one_value_l2956_295613


namespace equation_solutions_l2956_295689

/-- The equation we're solving -/
def equation (x : ℂ) : Prop :=
  x ≠ -2 ∧ (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48

/-- The set of solutions to the equation -/
def solutions : Set ℂ :=
  { x | x = 12 + 2*Real.sqrt 38 ∨
        x = 12 - 2*Real.sqrt 38 ∨
        x = -1/2 + Complex.I*(Real.sqrt 95)/2 ∨
        x = -1/2 - Complex.I*(Real.sqrt 95)/2 }

/-- Theorem stating that the solutions are correct and complete -/
theorem equation_solutions :
  ∀ x, equation x ↔ x ∈ solutions :=
sorry

end equation_solutions_l2956_295689


namespace theater_ticket_difference_l2956_295626

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def ticket_difference (ts : TicketSales) : ℕ :=
  let orchestra_tickets := (ts.total_revenue - ts.balcony_price * ts.total_tickets) / 
    (ts.orchestra_price - ts.balcony_price)
  let balcony_tickets := ts.total_tickets - orchestra_tickets
  balcony_tickets - orchestra_tickets

/-- Theorem stating the difference in ticket sales for the given scenario --/
theorem theater_ticket_difference :
  ∃ (ts : TicketSales), 
    ts.orchestra_price = 12 ∧
    ts.balcony_price = 8 ∧
    ts.total_tickets = 370 ∧
    ts.total_revenue = 3320 ∧
    ticket_difference ts = 190 := by
  sorry

end theater_ticket_difference_l2956_295626


namespace sum_and_equality_problem_l2956_295657

theorem sum_and_equality_problem (a b c : ℚ) :
  a + b + c = 120 ∧ (a + 8 = b - 3) ∧ (b - 3 = 3 * c) →
  b = 56 + 4/7 := by
sorry

end sum_and_equality_problem_l2956_295657


namespace count_valid_numbers_l2956_295600

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 100 ∧ n % sum_of_digits n = 0

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid n) ∧ S.card = 24 :=
sorry

end count_valid_numbers_l2956_295600


namespace seashells_given_correct_l2956_295659

/-- The number of seashells Tom gave to Jessica -/
def seashells_given : ℕ :=
  5 - 3

theorem seashells_given_correct : seashells_given = 2 := by
  sorry

end seashells_given_correct_l2956_295659


namespace pool_filling_time_l2956_295602

/-- Proves the time required to fill a pool given its volume and water delivery rates -/
theorem pool_filling_time 
  (pool_volume : ℝ) 
  (hose1_rate : ℝ) 
  (hose2_rate : ℝ) 
  (hose1_count : ℕ) 
  (hose2_count : ℕ) 
  (h1 : pool_volume = 15000) 
  (h2 : hose1_rate = 2) 
  (h3 : hose2_rate = 3) 
  (h4 : hose1_count = 2) 
  (h5 : hose2_count = 2) : 
  (pool_volume / (hose1_count * hose1_rate + hose2_count * hose2_rate)) / 60 = 25 := by
  sorry

#check pool_filling_time

end pool_filling_time_l2956_295602


namespace massager_vibration_rate_l2956_295695

theorem massager_vibration_rate (lowest_rate : ℝ) : 
  (∃ (highest_rate : ℝ),
    highest_rate = lowest_rate * 1.6 ∧ 
    (5 * 60) * highest_rate = 768000) →
  lowest_rate = 1600 := by
sorry

end massager_vibration_rate_l2956_295695


namespace unknown_blanket_rate_l2956_295639

/-- Given the purchase of blankets with known and unknown rates, prove the unknown rate -/
theorem unknown_blanket_rate (total_blankets : ℕ) (known_rate1 known_rate2 avg_rate : ℚ) 
  (count1 count2 count_unknown : ℕ) :
  total_blankets = count1 + count2 + count_unknown →
  count1 = 1 →
  count2 = 5 →
  count_unknown = 2 →
  known_rate1 = 100 →
  known_rate2 = 150 →
  avg_rate = 150 →
  (count1 * known_rate1 + count2 * known_rate2 + count_unknown * ((total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown)) / total_blankets = avg_rate →
  (total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown = 175 :=
by sorry

end unknown_blanket_rate_l2956_295639


namespace fractional_equation_solution_condition_l2956_295682

theorem fractional_equation_solution_condition (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ (m + x) / (2 - x) - 3 = 0) ↔ m ≠ -2 :=
by sorry

end fractional_equation_solution_condition_l2956_295682


namespace exists_non_unique_f_l2956_295655

theorem exists_non_unique_f : ∃ (f : ℕ → ℕ), 
  (∀ n : ℕ, f (f n) = 4 * n + 9) ∧ 
  (∀ k : ℕ, f (2^(k-1)) = 2^k + 3) ∧ 
  (∃ n : ℕ, f n ≠ 2 * n + 3) := by
  sorry

end exists_non_unique_f_l2956_295655


namespace units_digit_of_42_cubed_plus_24_cubed_l2956_295637

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_42_cubed_plus_24_cubed :
  unitsDigit (42^3 + 24^3) = 2 := by
  sorry


end units_digit_of_42_cubed_plus_24_cubed_l2956_295637


namespace cloth_cost_price_l2956_295658

/-- Given a cloth sale scenario, prove the cost price per meter -/
theorem cloth_cost_price
  (total_meters : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 35) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 70 := by
  sorry

end cloth_cost_price_l2956_295658


namespace line_equation_proof_l2956_295649

/-- A line passing through point (1, -2) with slope 3 has the equation 3x - y - 5 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - (-2) = 3 * (x - 1)) ↔ (3 * x - y - 5 = 0) := by sorry

end line_equation_proof_l2956_295649


namespace number_line_position_l2956_295601

/-- Given a number line where the distance from 0 to 25 is divided into 5 equal steps,
    the position after 4 steps from 0 is 20. -/
theorem number_line_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) :
  total_distance = 25 ∧ total_steps = 5 ∧ steps_taken = 4 →
  (total_distance / total_steps) * steps_taken = 20 := by
  sorry

end number_line_position_l2956_295601


namespace trigonometric_sum_equals_half_l2956_295681

theorem trigonometric_sum_equals_half : 
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by
  sorry

end trigonometric_sum_equals_half_l2956_295681


namespace train_speed_calculation_l2956_295622

/-- Proves that the speed of a train is approximately 80 km/hr given specific conditions -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) (man_speed_kmh : Real) :
  train_length = 220 →
  crossing_time = 10.999120070394369 →
  man_speed_kmh = 8 →
  ∃ (train_speed_kmh : Real), abs (train_speed_kmh - 80) < 0.1 := by
  sorry


end train_speed_calculation_l2956_295622


namespace integral_evaluation_l2956_295609

open Set
open MeasureTheory
open Interval

theorem integral_evaluation :
  ∫ x in (-2)..2, (x^2 * Real.sin x + Real.sqrt (4 - x^2)) = 2 * Real.pi :=
by
  have h1 : ∫ x in (-2)..2, x^2 * Real.sin x = 0 := sorry
  have h2 : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := sorry
  sorry

end integral_evaluation_l2956_295609


namespace smallest_integer_solution_minus_four_is_smallest_l2956_295612

theorem smallest_integer_solution (x : ℤ) : (7 - 3 * x < 22) ↔ (x ≥ -4) :=
  sorry

theorem minus_four_is_smallest : ∀ y : ℤ, (7 - 3 * y < 22) → y ≥ -4 :=
  sorry

end smallest_integer_solution_minus_four_is_smallest_l2956_295612


namespace function_satisfies_equation_l2956_295638

noncomputable def y (x : ℝ) : ℝ := Real.rpow (x - Real.log x - 1) (1/3)

theorem function_satisfies_equation (x : ℝ) (h : x > 0) :
  Real.log x + (y x)^3 - 3 * x * (y x)^2 * (deriv y x) = 0 := by
  sorry

end function_satisfies_equation_l2956_295638


namespace binomial_9_choose_3_l2956_295687

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end binomial_9_choose_3_l2956_295687


namespace kylie_coins_from_brother_l2956_295674

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

theorem kylie_coins_from_brother :
  piggy_bank_coins + brother_coins + father_coins - coins_given_away = coins_left :=
by sorry

end kylie_coins_from_brother_l2956_295674


namespace partner_A_share_l2956_295694

/-- Represents a partner's investment in a partnership --/
structure Investment where
  capital_ratio : ℚ
  time_ratio : ℚ

/-- Calculates the share of profit for a given investment --/
def calculate_share (inv : Investment) (total_capital_time : ℚ) (total_profit : ℚ) : ℚ :=
  (inv.capital_ratio * inv.time_ratio) / total_capital_time * total_profit

/-- Theorem stating that partner A's share of the profit is 100 --/
theorem partner_A_share :
  let a := Investment.mk (1/6) (1/6)
  let b := Investment.mk (1/3) (1/3)
  let c := Investment.mk (1/2) 1
  let total_capital_time := (1/6 * 1/6) + (1/3 * 1/3) + (1/2 * 1)
  let total_profit := 2300
  calculate_share a total_capital_time total_profit = 100 := by
  sorry

end partner_A_share_l2956_295694


namespace kayak_production_sum_l2956_295653

theorem kayak_production_sum (a : ℕ) (r : ℕ) (n : ℕ) : 
  a = 9 → r = 3 → n = 5 → a * (r^n - 1) / (r - 1) = 1089 := by
  sorry

end kayak_production_sum_l2956_295653


namespace intersection_of_P_and_Q_l2956_295685

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(4, -1)} := by sorry

end intersection_of_P_and_Q_l2956_295685


namespace berry_difference_l2956_295673

/-- The number of strawberries in a box -/
def strawberries_per_box : ℕ := 12

/-- The cost of a box of strawberries in dollars -/
def strawberry_box_cost : ℕ := 2

/-- The number of blueberries in a box -/
def blueberries_per_box : ℕ := 48

/-- The cost of a box of blueberries in dollars -/
def blueberry_box_cost : ℕ := 3

/-- The amount Sareen can spend in dollars -/
def sareen_budget : ℕ := 12

/-- The number of strawberries Sareen can buy -/
def m : ℕ := (sareen_budget / strawberry_box_cost) * strawberries_per_box

/-- The number of blueberries Sareen can buy -/
def n : ℕ := (sareen_budget / blueberry_box_cost) * blueberries_per_box

theorem berry_difference : n - m = 120 := by
  sorry

end berry_difference_l2956_295673


namespace living_room_count_l2956_295615

/-- The number of people in a house. -/
def total_people : ℕ := 15

/-- The number of people in the bedroom. -/
def bedroom_people : ℕ := 7

/-- The number of people in the living room. -/
def living_room_people : ℕ := total_people - bedroom_people

/-- Theorem stating that the number of people in the living room is 8. -/
theorem living_room_count : living_room_people = 8 := by
  sorry

end living_room_count_l2956_295615


namespace quadratic_form_sum_l2956_295628

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) → 
  a + h + k = -5/2 := by
  sorry

end quadratic_form_sum_l2956_295628


namespace map_scale_l2956_295677

/-- Given a map where 15 centimeters represents 90 kilometers,
    prove that 20 centimeters represents 120 kilometers. -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (length_cm : ℝ) :
  map_cm = 15 ∧ map_km = 90 ∧ length_cm = 20 →
  (length_cm / map_cm) * map_km = 120 := by
sorry

end map_scale_l2956_295677


namespace directrix_of_given_parabola_l2956_295641

/-- A parabola with equation y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- The parabola y = 4x^2 - 3 -/
def given_parabola : Parabola := { a := 4, b := -3 }

theorem directrix_of_given_parabola :
  directrix given_parabola = -19/16 := by sorry

end directrix_of_given_parabola_l2956_295641


namespace unique_intersection_implies_a_equals_three_l2956_295686

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 2

def intersection_count (f g : ℝ → ℝ) : ℕ := sorry

theorem unique_intersection_implies_a_equals_three :
  ∀ a : ℝ, intersection_count f (g a) = 1 → a = 3 := by sorry

end unique_intersection_implies_a_equals_three_l2956_295686


namespace chess_tournament_schedules_l2956_295646

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of games each player plays against each opponent -/
def games_per_opponent : ℕ := 3

/-- Represents the number of games played simultaneously in each round -/
def games_per_round : ℕ := 3

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players * games_per_opponent

/-- Calculates the number of rounds in the tournament -/
def num_rounds : ℕ := total_games / games_per_round

/-- Theorem stating the number of distinct ways to schedule the tournament -/
theorem chess_tournament_schedules :
  (Nat.factorial num_rounds) / (Nat.factorial games_per_round) =
  (Nat.factorial 16) / (Nat.factorial 3) :=
sorry

end chess_tournament_schedules_l2956_295646


namespace second_divisor_problem_l2956_295647

theorem second_divisor_problem (initial : ℝ) (first_divisor : ℝ) (final_result : ℝ) (x : ℝ) :
  initial = 8900 →
  first_divisor = 6 →
  final_result = 370.8333333333333 →
  (initial / first_divisor) / x = final_result →
  x = 4 := by
  sorry

end second_divisor_problem_l2956_295647


namespace jake_papayas_l2956_295699

/-- The number of papayas Jake's brother can eat in one week -/
def brother_papayas : ℕ := 5

/-- The number of papayas Jake's father can eat in one week -/
def father_papayas : ℕ := 4

/-- The total number of papayas needed for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- Theorem: Jake can eat 3 papayas in one week -/
theorem jake_papayas : 
  ∃ (j : ℕ), j = 3 ∧ num_weeks * (j + brother_papayas + father_papayas) = total_papayas :=
by sorry

end jake_papayas_l2956_295699


namespace max_annual_profit_l2956_295675

noncomputable section

def fixed_cost : ℝ := 2.6

def additional_investment (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

def selling_price : ℝ := 0.9

def annual_profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then selling_price * x - additional_investment x - fixed_cost
  else (selling_price * x * x - (901 * x^2 - 9450 * x + 10000)) / x - fixed_cost

theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ annual_profit x = 8990 ∧
  ∀ (y : ℝ), y ≥ 0 → annual_profit y ≤ annual_profit x :=
sorry

end max_annual_profit_l2956_295675


namespace amount_after_two_years_l2956_295669

/-- The annual growth rate -/
def r : ℚ := 1 / 8

/-- The initial amount -/
def initial_amount : ℚ := 76800

/-- The amount after n years -/
def amount_after (n : ℕ) : ℚ := initial_amount * (1 + r) ^ n

/-- Theorem: The amount after two years is 97200 -/
theorem amount_after_two_years : amount_after 2 = 97200 := by
  sorry

end amount_after_two_years_l2956_295669


namespace marcos_strawberry_weight_l2956_295643

/-- Marco and his dad went strawberry picking. This theorem proves the weight of Marco's strawberries. -/
theorem marcos_strawberry_weight
  (total_weight : ℕ)
  (dads_weight : ℕ)
  (h1 : total_weight = 40)
  (h2 : dads_weight = 32)
  : total_weight - dads_weight = 8 := by
  sorry

end marcos_strawberry_weight_l2956_295643


namespace log_27_3_l2956_295620

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end log_27_3_l2956_295620


namespace triangle_existence_uniqueness_l2956_295698

/-- A point in 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Incircle of a triangle -/
structure Incircle where
  center : Point
  radius : ℝ

/-- Excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point lies on a line segment -/
def lies_on_segment (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is the tangency point of a circle and a line -/
def is_tangency_point (P : Point) (C : Incircle ⊕ Excircle) (L M : Point) : Prop := sorry

/-- Theorem stating the existence and uniqueness of a triangle given specific tangency points -/
theorem triangle_existence_uniqueness 
  (T_a T_aa T_c T_ac : Point) 
  (h_distinct : T_a ≠ T_aa ∧ T_c ≠ T_ac) 
  (h_not_collinear : ¬ lies_on_segment T_a T_c T_aa) : 
  ∃! (ABC : Triangle) (k : Incircle) (k' : Excircle), 
    is_tangency_point T_a (Sum.inl k) ABC.B ABC.C ∧ 
    is_tangency_point T_aa (Sum.inr k') ABC.B ABC.C ∧
    is_tangency_point T_c (Sum.inl k) ABC.A ABC.B ∧
    is_tangency_point T_ac (Sum.inr k') ABC.A ABC.B := by
  sorry

end triangle_existence_uniqueness_l2956_295698


namespace number_problem_l2956_295630

theorem number_problem (x y : ℝ) 
  (h1 : (40 / 100) * x = (30 / 100) * 50)
  (h2 : (60 / 100) * x = (45 / 100) * y) :
  x = 37.5 ∧ y = 50 := by
  sorry

end number_problem_l2956_295630


namespace school_supplies_cost_l2956_295650

/-- Calculates the total cost of school supplies with discounts applied --/
theorem school_supplies_cost 
  (haley_paper_price : ℝ) 
  (haley_paper_quantity : ℕ)
  (sister_paper_price : ℝ)
  (sister_paper_quantity : ℕ)
  (paper_discount : ℝ)
  (haley_pen_price : ℝ)
  (haley_pen_quantity : ℕ)
  (sister_pen_price : ℝ)
  (sister_pen_quantity : ℕ)
  (pen_discount : ℝ)
  (h1 : haley_paper_price = 3.75)
  (h2 : haley_paper_quantity = 2)
  (h3 : sister_paper_price = 4.50)
  (h4 : sister_paper_quantity = 3)
  (h5 : paper_discount = 0.5)
  (h6 : haley_pen_price = 1.45)
  (h7 : haley_pen_quantity = 5)
  (h8 : sister_pen_price = 1.65)
  (h9 : sister_pen_quantity = 7)
  (h10 : pen_discount = 0.25)
  : ℝ := by
  sorry

end school_supplies_cost_l2956_295650


namespace hidden_dots_sum_l2956_295680

/-- The sum of numbers on a single die --/
def single_die_sum : ℕ := 21

/-- The total number of dice --/
def total_dice : ℕ := 4

/-- The number of visible faces --/
def visible_faces : ℕ := 10

/-- The sum of visible numbers --/
def visible_sum : ℕ := 37

theorem hidden_dots_sum :
  single_die_sum * total_dice - visible_sum = 47 := by
  sorry

end hidden_dots_sum_l2956_295680


namespace area_of_PRQ_l2956_295636

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15)
  (xz_length : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 14)
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 7)

-- Define the circumcenter P
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter Q
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the point R
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the condition for R being tangent to XZ, YZ, and the circumcircle
def is_tangent (t : Triangle) (r : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_PRQ (t : Triangle) 
  (h : is_tangent t (R t)) : 
  triangle_area (circumcenter t) (incenter t) (R t) = 245 / 72 := by
  sorry

end area_of_PRQ_l2956_295636


namespace fraction_problem_l2956_295606

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4 : ℝ) * (1/3 : ℝ) * F * N = 35)
  (h2 : (40/100 : ℝ) * N = 420) : 
  F = 2/5 := by
sorry

end fraction_problem_l2956_295606


namespace batsman_average_is_37_l2956_295697

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  averageIncrease : ℕ
  lastInningScore : ℕ

/-- Calculates the new average score after the latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

/-- Theorem: Given the conditions, prove that the new average is 37 -/
theorem batsman_average_is_37 (b : Batsman)
    (h1 : b.innings = 17)
    (h2 : b.lastInningScore = 85)
    (h3 : b.averageIncrease = 3)
    (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
    newAverage b = 37 := by
  sorry

end batsman_average_is_37_l2956_295697


namespace max_min_values_of_f_l2956_295640

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem max_min_values_of_f :
  (∃ (x : ℝ), x ∈ I ∧ f x = 5) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 5) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ 1) :=
by sorry

end max_min_values_of_f_l2956_295640


namespace large_square_area_l2956_295660

theorem large_square_area (s : ℝ) (h1 : s > 0) (h2 : 2 * s^2 = 14) : (3 * s)^2 = 63 := by
  sorry

end large_square_area_l2956_295660


namespace line_intercepts_sum_l2956_295688

theorem line_intercepts_sum (c : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + c = 0 ∧ x + y = 30) → c = -56.25 := by
  sorry

end line_intercepts_sum_l2956_295688


namespace no_roots_composition_l2956_295645

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_composition (b c : ℝ) 
  (h : ∀ x : ℝ, f b c x ≠ x) : 
  ∀ x : ℝ, f b c (f b c x) ≠ x := by
  sorry

end no_roots_composition_l2956_295645


namespace families_without_pets_l2956_295652

theorem families_without_pets (total : ℕ) (cats : ℕ) (dogs : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : cats = 18)
  (h3 : dogs = 24)
  (h4 : both = 10) :
  total - (cats + dogs - both) = 8 := by
  sorry

end families_without_pets_l2956_295652


namespace intersection_of_A_and_B_l2956_295616

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l2956_295616
