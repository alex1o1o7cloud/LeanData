import Mathlib

namespace NUMINAMATH_CALUDE_no_common_points_l4068_406842

theorem no_common_points (m : ℝ) : 
  (∀ x y : ℝ, x + m^2 * y + 6 = 0 ∧ (m - 2) * x + 3 * m * y + 2 * m = 0 → False) ↔ 
  (m = 0 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_no_common_points_l4068_406842


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4068_406801

theorem perpendicular_vectors (m : ℝ) : 
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (m, 1)
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → m = 7 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4068_406801


namespace NUMINAMATH_CALUDE_closest_irrational_to_four_l4068_406889

theorem closest_irrational_to_four :
  let options : List ℝ := [Real.sqrt 11, Real.sqrt 13, Real.sqrt 17, Real.sqrt 19]
  let four : ℝ := Real.sqrt 16
  ∀ x ∈ options, x ≠ Real.sqrt 17 →
    |Real.sqrt 17 - four| < |x - four| :=
by sorry

end NUMINAMATH_CALUDE_closest_irrational_to_four_l4068_406889


namespace NUMINAMATH_CALUDE_no_five_integers_with_prime_triples_l4068_406892

theorem no_five_integers_with_prime_triples : ¬ ∃ (a b c d e : ℕ+),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (∀ (x y z : ℕ+), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                   (x ≠ y ∧ x ≠ z ∧ y ≠ z) →
                   Nat.Prime (x.val + y.val + z.val)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_integers_with_prime_triples_l4068_406892


namespace NUMINAMATH_CALUDE_discount_percentage_l4068_406847

theorem discount_percentage (tshirt_cost pants_cost shoes_cost : ℝ)
  (tshirt_qty pants_qty shoes_qty : ℕ)
  (total_paid : ℝ)
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : shoes_cost = 150)
  (h4 : tshirt_qty = 4)
  (h5 : pants_qty = 3)
  (h6 : shoes_qty = 2)
  (h7 : total_paid = 558) :
  (1 - total_paid / (tshirt_cost * tshirt_qty + pants_cost * pants_qty + shoes_cost * shoes_qty)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l4068_406847


namespace NUMINAMATH_CALUDE_sum_of_abc_l4068_406859

theorem sum_of_abc (a b c : ℝ) : 
  a * (a - 4) = 5 →
  b * (b - 4) = 5 →
  c * (c - 4) = 5 →
  a^2 + b^2 = c^2 →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  a + b + c = 4 + Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l4068_406859


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l4068_406881

/-- A line passing through the point (2, 1) with a slope of 2 has the equation 2x - y - 3 = 0. -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (2 : ℝ) * x - y - 3 = 0 ↔ (y - 1 = 2 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l4068_406881


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l4068_406833

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l4068_406833


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l4068_406838

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l4068_406838


namespace NUMINAMATH_CALUDE_sourball_theorem_l4068_406848

def sourball_problem (nellie jacob lana bucket_total : ℕ) : Prop :=
  nellie = 12 ∧
  jacob = nellie / 2 ∧
  lana = jacob - 3 ∧
  bucket_total = 30 ∧
  let total_eaten := nellie + jacob + lana
  let remaining := bucket_total - total_eaten
  remaining / 3 = 3

theorem sourball_theorem :
  ∃ (nellie jacob lana bucket_total : ℕ),
    sourball_problem nellie jacob lana bucket_total :=
by
  sorry

end NUMINAMATH_CALUDE_sourball_theorem_l4068_406848


namespace NUMINAMATH_CALUDE_power_division_equality_l4068_406831

theorem power_division_equality : 3^12 / 27^2 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l4068_406831


namespace NUMINAMATH_CALUDE_product_equals_zero_l4068_406804

theorem product_equals_zero (b : ℤ) (h : b = 5) : 
  ((b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
   (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_equals_zero_l4068_406804


namespace NUMINAMATH_CALUDE_extreme_value_odd_function_l4068_406854

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

-- State the theorem
theorem extreme_value_odd_function 
  (a b c : ℝ) 
  (h1 : f a b c 1 = c - 4)  -- f(x) reaches c-4 at x=1
  (h2 : ∀ x, f a b c (-x) = -(f a b c x))  -- f(x) is odd
  : 
  (a = 2 ∧ b = -6) ∧  -- Part 1: values of a and b
  (∀ x ∈ Set.Ioo (-2) 0, f a b c x ≤ 4)  -- Part 2: maximum value on (-2,0)
  :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_odd_function_l4068_406854


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l4068_406805

theorem sum_of_four_numbers : 5678 + 6785 + 7856 + 8567 = 28886 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l4068_406805


namespace NUMINAMATH_CALUDE_house_renovation_time_l4068_406832

theorem house_renovation_time :
  let num_bedrooms : ℕ := 3
  let bedroom_time : ℕ := 4
  let kitchen_time : ℕ := bedroom_time + bedroom_time / 2
  let bedrooms_and_kitchen_time : ℕ := num_bedrooms * bedroom_time + kitchen_time
  let living_room_time : ℕ := 2 * bedrooms_and_kitchen_time
  let total_time : ℕ := bedrooms_and_kitchen_time + living_room_time
  total_time = 54 := by sorry

end NUMINAMATH_CALUDE_house_renovation_time_l4068_406832


namespace NUMINAMATH_CALUDE_hotel_beds_count_l4068_406863

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : beds_in_two_bed_room = 2)
  (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 :=
by sorry

end NUMINAMATH_CALUDE_hotel_beds_count_l4068_406863


namespace NUMINAMATH_CALUDE_average_salary_proof_l4068_406869

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 15000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_individuals : ℕ := 5

theorem average_salary_proof :
  total_salary / num_individuals = 9000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l4068_406869


namespace NUMINAMATH_CALUDE_debby_water_bottles_l4068_406874

/-- The number of water bottles Debby drinks per day -/
def bottles_per_day : ℕ := 6

/-- The number of days the water bottles would last -/
def days_lasting : ℕ := 2

/-- The number of water bottles Debby bought -/
def bottles_bought : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : bottles_bought = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l4068_406874


namespace NUMINAMATH_CALUDE_range_of_m_l4068_406820

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (7*m - 3) = 1 ∧ 
  (m + 3 > 0) ∧ (7*m - 3 < 0)

def q (m : ℝ) : Prop := ∀ (x : ℝ), Monotone (fun x => (5 - 2*m)^x)

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) → 
  (∀ m : ℝ, (m ≤ -3 ∨ (3/7 ≤ m ∧ m < 2))) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l4068_406820


namespace NUMINAMATH_CALUDE_max_x_value_l4068_406844

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 10) :
  x ≤ 2 ∧ ∃ y z, x = 2 ∧ y + z = 4 ∧ x + y + z = 6 ∧ x*y + x*z + y*z = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l4068_406844


namespace NUMINAMATH_CALUDE_factorial_difference_l4068_406884

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l4068_406884


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l4068_406899

/-- The equation x^2 - 18y^2 - 6x + 4y + 9 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), x^2 - 18*y^2 - 6*x + 4*y + 9 = 0 ↔
  ((x - c) / a)^2 - ((y - d) / b)^2 = 1 ∧
  e = 1 := by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l4068_406899


namespace NUMINAMATH_CALUDE_quartic_polynomial_value_l4068_406835

/-- A quartic polynomial with specific properties -/
def QuarticPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧ 
  P 1 = 0 ∧
  (∀ x, P x ≤ 3) ∧
  P 2 = 3 ∧
  P 3 = 3

/-- The main theorem -/
theorem quartic_polynomial_value (P : ℝ → ℝ) (h : QuarticPolynomial P) : P 5 = -24 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_value_l4068_406835


namespace NUMINAMATH_CALUDE_parabola_directrix_l4068_406823

/-- The directrix of the parabola y² = 4x is the line x = -1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (a : ℝ), a = -1 ∧ x = a) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4068_406823


namespace NUMINAMATH_CALUDE_range_of_a_l4068_406898

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4068_406898


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l4068_406883

/-- Given a line y = mx - 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    the possible slopes m satisfy m^2 ≥ 4/41 -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x - 3) → m^2 ≥ 4/41 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l4068_406883


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l4068_406865

/-- Represents the colors of flags -/
inductive FlagColor
  | Blue
  | Red
  | Green

/-- Represents a pair of flags -/
structure FlagPair where
  first : FlagColor
  second : FlagColor

/-- The distribution of flag pairs among children -/
structure FlagDistribution where
  blueRed : ℚ
  redGreen : ℚ
  blueGreen : ℚ
  allThree : ℚ

/-- The problem statement -/
theorem flag_distribution_theorem (dist : FlagDistribution) :
  dist.blueRed = 1/2 →
  dist.redGreen = 3/10 →
  dist.blueGreen = 1/10 →
  dist.allThree = 1/10 →
  dist.blueRed + dist.redGreen + dist.blueGreen + dist.allThree = 1 →
  (dist.blueRed + dist.redGreen + dist.blueGreen - dist.allThree + dist.allThree : ℚ) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l4068_406865


namespace NUMINAMATH_CALUDE_jason_total_cards_l4068_406830

/-- The number of Pokemon cards Jason has after receiving new ones from Alyssa -/
def total_cards (initial_cards new_cards : ℕ) : ℕ :=
  initial_cards + new_cards

/-- Theorem stating that Jason's total cards is 900 given the initial and new card counts -/
theorem jason_total_cards :
  total_cards 676 224 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_cards_l4068_406830


namespace NUMINAMATH_CALUDE_floor_times_self_equals_72_l4068_406815

theorem floor_times_self_equals_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_72_l4068_406815


namespace NUMINAMATH_CALUDE_extrema_of_f_l4068_406817

def f (x : ℝ) := -x^2 + x + 1

theorem extrema_of_f :
  let a := 0
  let b := 3/2
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc a b ∧ x_max ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    f x_min = 1/4 ∧ f x_max = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_extrema_of_f_l4068_406817


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4068_406814

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x^2 + 1 / x^6 ≥ 6 * Real.sqrt 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 9 * x^2 + 1 / x^6 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4068_406814


namespace NUMINAMATH_CALUDE_marys_nickels_l4068_406893

/-- Given Mary's initial nickels and the additional nickels from her dad,
    calculate the total number of nickels Mary has now. -/
theorem marys_nickels (initial : ℕ) (additional : ℕ) : 
  initial = 7 → additional = 5 → initial + additional = 12 := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l4068_406893


namespace NUMINAMATH_CALUDE_expression_evaluation_l4068_406896

theorem expression_evaluation :
  Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4068_406896


namespace NUMINAMATH_CALUDE_ancient_chinese_gcd_is_successive_differences_l4068_406880

/-- The algorithm used by ancient Chinese mathematicians to find the GCD of two positive integers -/
def ancient_chinese_gcd_algorithm : Type := sorry

/-- The method of successive differences -/
def successive_differences : Type := sorry

/-- Assertion that the ancient Chinese GCD algorithm is the method of successive differences -/
theorem ancient_chinese_gcd_is_successive_differences : 
  ancient_chinese_gcd_algorithm = successive_differences := by sorry

end NUMINAMATH_CALUDE_ancient_chinese_gcd_is_successive_differences_l4068_406880


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l4068_406828

/-- A circle C with center (1,2) that is tangent to the line x+2y=0 -/
def CircleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5}

/-- The line x+2y=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2*p.2 = 0}

theorem circle_equation_is_correct :
  (∀ p ∈ CircleC, (p.1 - 1)^2 + (p.2 - 2)^2 = 5) ∧
  (∃ q ∈ CircleC ∩ TangentLine, q = (1, 2)) ∧
  (∀ r ∈ CircleC, r ≠ (1, 2) → r ∉ TangentLine) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l4068_406828


namespace NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_480_l4068_406802

def S : Finset Int := {-10, -7, -3, 1, 4, 6, 8}

theorem min_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z :=
by
  sorry

theorem min_product_is_neg_480 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = -480 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z →
   a * b * c ≤ x * y * z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_480_l4068_406802


namespace NUMINAMATH_CALUDE_sally_lost_balloons_l4068_406887

theorem sally_lost_balloons (initial_orange : ℕ) (current_orange : ℕ) 
  (h1 : initial_orange = 9) 
  (h2 : current_orange = 7) : 
  initial_orange - current_orange = 2 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_balloons_l4068_406887


namespace NUMINAMATH_CALUDE_floor_a_n_l4068_406839

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

/-- Theorem stating that the floor of a_n is 1994 - n for 0 ≤ n ≤ 998 -/
theorem floor_a_n (n : ℕ) (h : n ≤ 998) : 
  ⌊a n⌋ = 1994 - n := by sorry

end NUMINAMATH_CALUDE_floor_a_n_l4068_406839


namespace NUMINAMATH_CALUDE_vector_equality_l4068_406871

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

theorem vector_equality (x : ℝ) : a x = (B.1 - A.1, B.2 - A.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l4068_406871


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4068_406829

theorem inscribed_circle_radius (a b c r : ℝ) : 
  a = 6 → b = 12 → c = 18 → 
  (1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))) →
  r = 36 / 17 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4068_406829


namespace NUMINAMATH_CALUDE_no_real_solutions_l4068_406878

theorem no_real_solutions : ¬∃ (x : ℝ), (2*x - 3*x + 7)^2 + 4 = -|2*x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4068_406878


namespace NUMINAMATH_CALUDE_n_value_l4068_406875

theorem n_value (e n : ℕ+) : 
  (Nat.lcm e n = 690) →
  (100 ≤ n) →
  (n < 1000) →
  (¬ 3 ∣ n) →
  (¬ 2 ∣ e) →
  (n = 230) :=
sorry

end NUMINAMATH_CALUDE_n_value_l4068_406875


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_l4068_406808

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where

/-- The number of faces in a rectangular prism. -/
def faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism. -/
def vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26. -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  faces rp + edges rp + vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_l4068_406808


namespace NUMINAMATH_CALUDE_expression_evaluation_l4068_406895

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  (3 * x + 2 * y) - (3 * x - 2 * y) = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4068_406895


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l4068_406812

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101011 : List Bool :=
  [true, false, true, true, false, true, false, true, true]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_101101011) = [3, 2, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l4068_406812


namespace NUMINAMATH_CALUDE_quarterly_to_annual_compound_interest_l4068_406857

/-- Given an annual interest rate of 8% compounded quarterly, 
    prove that it's equivalent to an 8.24% annual rate compounded annually. -/
theorem quarterly_to_annual_compound_interest : 
  let quarterly_rate : ℝ := 0.08 / 4
  let effective_annual_rate : ℝ := (1 + quarterly_rate)^4 - 1
  ∀ ε > 0, |effective_annual_rate - 0.0824| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_quarterly_to_annual_compound_interest_l4068_406857


namespace NUMINAMATH_CALUDE_male_listeners_count_l4068_406862

/-- Represents the survey data for Radio Wave XFM --/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey data --/
def male_listeners (data : SurveyData) : ℕ :=
  data.total_listeners - data.female_listeners

/-- Theorem stating that the number of male listeners is 75 --/
theorem male_listeners_count (data : SurveyData)
  (h1 : data.total_listeners = 150)
  (h2 : data.total_non_listeners = 180)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 84) :
  male_listeners data = 75 := by
  sorry

#eval male_listeners { total_listeners := 150, total_non_listeners := 180, female_listeners := 75, male_non_listeners := 84 }

end NUMINAMATH_CALUDE_male_listeners_count_l4068_406862


namespace NUMINAMATH_CALUDE_expression_evaluation_l4068_406836

theorem expression_evaluation : 
  (2024^3 - 3 * 2024^2 * 2025 + 4 * 2024 * 2025^2 - 2025^3 + 2) / (2024 * 2025) = 2025 - 1 / (2024 * 2025) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4068_406836


namespace NUMINAMATH_CALUDE_symmetry_implies_periodic_l4068_406813

/-- A function that is symmetric with respect to two distinct points is periodic. -/
theorem symmetry_implies_periodic (f : ℝ → ℝ) (a b : ℝ) 
  (ha : ∀ x, f (a - x) = f (a + x))
  (hb : ∀ x, f (b - x) = f (b + x))
  (hab : a ≠ b) :
  ∀ x, f (x + (2*b - 2*a)) = f x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodic_l4068_406813


namespace NUMINAMATH_CALUDE_fraction_simplification_l4068_406897

theorem fraction_simplification (c d : ℝ) : 
  (5 + 4 * c - 3 * d) / 9 + 5 = (50 + 4 * c - 3 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4068_406897


namespace NUMINAMATH_CALUDE_eighth_term_is_sixteen_l4068_406816

def odd_term (n : ℕ) : ℕ := 2 * n - 1

def even_term (n : ℕ) : ℕ := 4 * n

def sequence_term (n : ℕ) : ℕ :=
  if n % 2 = 1 then odd_term ((n + 1) / 2) else even_term (n / 2)

theorem eighth_term_is_sixteen : sequence_term 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_sixteen_l4068_406816


namespace NUMINAMATH_CALUDE_additional_teddies_calculation_l4068_406837

/-- The number of additional teddies Jina gets for each bunny -/
def additional_teddies_per_bunny : ℕ :=
  let initial_teddies : ℕ := 5
  let bunnies : ℕ := 3 * initial_teddies
  let koalas : ℕ := 1
  let total_mascots : ℕ := 51
  let initial_mascots : ℕ := initial_teddies + bunnies + koalas
  let additional_teddies : ℕ := total_mascots - initial_mascots
  additional_teddies / bunnies

theorem additional_teddies_calculation : additional_teddies_per_bunny = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_teddies_calculation_l4068_406837


namespace NUMINAMATH_CALUDE_power_division_l4068_406849

theorem power_division : 2^12 / 8^3 = 8 := by sorry

end NUMINAMATH_CALUDE_power_division_l4068_406849


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_lines_l4068_406864

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (non_overlapping_planes : Plane → Plane → Prop)
variable (non_overlapping_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_lines
  (α β : Plane) (l m : Line)
  (h1 : non_overlapping_planes α β)
  (h2 : non_overlapping_lines l m)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : line_parallel l m) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_lines_l4068_406864


namespace NUMINAMATH_CALUDE_sheet_width_l4068_406894

theorem sheet_width (w : ℝ) 
  (h1 : w > 0)
  (h2 : (w - 4) * 24 / (w * 30) = 64 / 100) : 
  w = 20 := by sorry

end NUMINAMATH_CALUDE_sheet_width_l4068_406894


namespace NUMINAMATH_CALUDE_area_ratio_in_equally_divided_perimeter_l4068_406819

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A point on the perimeter of a triangle -/
structure PerimeterPoint (t : Triangle) where
  point : ℝ × ℝ
  on_perimeter : sorry

/-- Theorem: Area ratio in a triangle with equally divided perimeter -/
theorem area_ratio_in_equally_divided_perimeter (ABC : Triangle) 
  (P Q R : PerimeterPoint ABC) : 
  perimeter ABC = 1 →
  P.point.1 < Q.point.1 →
  (P.point.1 - ABC.A.1) + (Q.point.1 - P.point.1) + 
    (perimeter ABC - (Q.point.1 - ABC.A.1)) = perimeter ABC →
  let PQR : Triangle := ⟨P.point, Q.point, R.point⟩
  area PQR / area ABC > 2/9 := by sorry

end NUMINAMATH_CALUDE_area_ratio_in_equally_divided_perimeter_l4068_406819


namespace NUMINAMATH_CALUDE_product_minus_quotient_l4068_406861

theorem product_minus_quotient : 11 * 13 * 17 - 33 / 3 = 2420 := by
  sorry

end NUMINAMATH_CALUDE_product_minus_quotient_l4068_406861


namespace NUMINAMATH_CALUDE_pizza_distribution_l4068_406890

/-- Given 12 coworkers sharing 3 pizzas with 8 slices each, prove that each person gets 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l4068_406890


namespace NUMINAMATH_CALUDE_table_area_proof_l4068_406818

theorem table_area_proof (total_runner_area : ℝ) (coverage_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 224) 
  (h2 : coverage_percentage = 0.8)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 30) : 
  ∃ (table_area : ℝ), table_area = 175 ∧ 
    coverage_percentage * table_area = 
      (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
      two_layer_area + three_layer_area := by
  sorry

end NUMINAMATH_CALUDE_table_area_proof_l4068_406818


namespace NUMINAMATH_CALUDE_sequence_term_from_sum_l4068_406803

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 + 3*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_term_from_sum (n : ℕ) : 
  n > 0 → S n - S (n-1) = a n :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_from_sum_l4068_406803


namespace NUMINAMATH_CALUDE_greatest_circle_center_distance_l4068_406891

theorem greatest_circle_center_distance
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        (x₁ - circle_diameter / 2 ≥ 0) ∧
        (y₁ - circle_diameter / 2 ≥ 0) ∧
        (x₁ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₁ + circle_diameter / 2 ≤ rectangle_height) ∧
        (x₂ - circle_diameter / 2 ≥ 0) ∧
        (y₂ - circle_diameter / 2 ≥ 0) ∧
        (x₂ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₂ + circle_diameter / 2 ≤ rectangle_height) ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_circle_center_distance_l4068_406891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4068_406858

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4068_406858


namespace NUMINAMATH_CALUDE_perpendicular_vectors_second_component_l4068_406870

/-- Given two 2D vectors a and b, if they are perpendicular, then the second component of b is 2. -/
theorem perpendicular_vectors_second_component (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  b.2 = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_second_component_l4068_406870


namespace NUMINAMATH_CALUDE_second_polygon_sides_l4068_406879

theorem second_polygon_sides (p1 p2 : ℕ) (s : ℝ) :
  p1 = 50 →                          -- First polygon has 50 sides
  p1 * (3 * s) = p2 * s →            -- Same perimeter
  3 * s > 0 →                        -- Positive side length
  p2 = 150 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l4068_406879


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l4068_406868

/-- The number of peaches Sally picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l4068_406868


namespace NUMINAMATH_CALUDE_angle_A_value_l4068_406843

theorem angle_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_l4068_406843


namespace NUMINAMATH_CALUDE_sum_remainder_l4068_406853

theorem sum_remainder (x y : ℤ) (hx : x % 72 = 19) (hy : y % 50 = 6) : 
  (x + y) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l4068_406853


namespace NUMINAMATH_CALUDE_product_b_sample_size_l4068_406822

/-- Represents the number of items drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (totalItems : ℕ) (sampleSize : ℕ) (groupRatio : ℕ) (totalRatio : ℕ) : ℕ :=
  (sampleSize * groupRatio) / totalRatio

/-- Proves that the number of items drawn from product B in the given stratified sampling scenario is 10 -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let sampleSize : ℕ := 60
  let ratioA : ℕ := 1
  let ratioB : ℕ := 2
  let ratioC : ℕ := 4
  let ratioD : ℕ := 5
  let totalRatio : ℕ := ratioA + ratioB + ratioC + ratioD
  stratifiedSampleSize totalItems sampleSize ratioB totalRatio = 10 := by
  sorry


end NUMINAMATH_CALUDE_product_b_sample_size_l4068_406822


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_t_value_l4068_406855

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for 2D vectors -/
def collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem vector_collinearity_implies_t_value :
  let OA : Vector2D := ⟨1, 2⟩
  let OB : Vector2D := ⟨3, 4⟩
  let OC : Vector2D := ⟨2*t, t+5⟩
  let AB : Vector2D := ⟨OB.x - OA.x, OB.y - OA.y⟩
  let AC : Vector2D := ⟨OC.x - OA.x, OC.y - OA.y⟩
  collinear AB AC → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_t_value_l4068_406855


namespace NUMINAMATH_CALUDE_scarlett_oil_addition_l4068_406821

/-- The amount of oil Scarlett needs to add to her measuring cup -/
def oil_to_add (initial_amount final_amount : ℚ) : ℚ :=
  final_amount - initial_amount

/-- Theorem stating the amount of oil Scarlett needs to add -/
theorem scarlett_oil_addition (initial_amount final_amount : ℚ) 
  (h1 : initial_amount = 17/100)
  (h2 : final_amount = 84/100) :
  oil_to_add initial_amount final_amount = 67/100 := by
  sorry

#eval oil_to_add (17/100) (84/100)

end NUMINAMATH_CALUDE_scarlett_oil_addition_l4068_406821


namespace NUMINAMATH_CALUDE_correct_average_marks_l4068_406867

/-- Calculates the correct average marks for a class given the following conditions:
  * There are 40 students in the class
  * The reported average marks are 65
  * Three students' marks were wrongly noted:
    - First student: 100 instead of 20
    - Second student: 85 instead of 50
    - Third student: 15 instead of 55
-/
theorem correct_average_marks (num_students : ℕ) (reported_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ) :
  num_students = 40 →
  reported_average = 65 →
  incorrect_mark1 = 100 →
  incorrect_mark2 = 85 →
  incorrect_mark3 = 15 →
  correct_mark1 = 20 →
  correct_mark2 = 50 →
  correct_mark3 = 55 →
  (num_students * reported_average - (incorrect_mark1 + incorrect_mark2 + incorrect_mark3) +
    (correct_mark1 + correct_mark2 + correct_mark3)) / num_students = 63125 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l4068_406867


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4068_406851

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (a * b) ^ 2 = a * c * b * c →  -- geometric sequence
  a + b + c = 6 →  -- sum condition
  a = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4068_406851


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4068_406860

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x ≤ 22 ∧ (1053 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1053 + y) % 23 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4068_406860


namespace NUMINAMATH_CALUDE_coin_game_expected_value_l4068_406800

/-- A modified coin game with three outcomes --/
structure CoinGame where
  prob_heads : ℝ
  prob_tails : ℝ
  prob_edge : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  payoff_edge : ℝ

/-- Calculate the expected value of the coin game --/
def expected_value (game : CoinGame) : ℝ :=
  game.prob_heads * game.payoff_heads +
  game.prob_tails * game.payoff_tails +
  game.prob_edge * game.payoff_edge

/-- Theorem stating the expected value of the specific coin game --/
theorem coin_game_expected_value :
  let game : CoinGame := {
    prob_heads := 1/4,
    prob_tails := 1/2,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value game = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_coin_game_expected_value_l4068_406800


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l4068_406825

theorem cube_sum_theorem (x y k c : ℝ) (h1 : x^3 * y^3 = k) (h2 : 1 / x^3 + 1 / y^3 = c) :
  ∃ m : ℝ, m = x + y ∧ (x + y)^3 = c * k + 3 * (k^(1/3)) * m :=
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l4068_406825


namespace NUMINAMATH_CALUDE_magnitude_of_z_l4068_406886

theorem magnitude_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l4068_406886


namespace NUMINAMATH_CALUDE_second_player_wins_l4068_406845

/-- Represents the game board -/
def Board := Fin 3 → Fin 101 → Bool

/-- The initial state of the board with the central cell crossed out -/
def initialBoard : Board :=
  fun i j => i = 1 && j = 50

/-- A move in the game -/
structure Move where
  start_row : Fin 3
  start_col : Fin 101
  length : Fin 4
  direction : Bool  -- true for down-right, false for down-left

/-- Checks if a move is valid on the given board -/
def isValidMove (b : Board) (m : Move) : Bool :=
  sorry

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (b : Board) : Bool :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Board → Move),
    ∀ (game : List Move),
      game.length % 2 = 0 →
      let final_board := game.foldl applyMove initialBoard
      isGameOver final_board →
      ∃ (m : Move), isValidMove final_board m :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l4068_406845


namespace NUMINAMATH_CALUDE_cylinder_height_after_forging_l4068_406877

theorem cylinder_height_after_forging (initial_diameter initial_height new_diameter : ℝ) 
  (h_initial_diameter : initial_diameter = 6)
  (h_initial_height : initial_height = 24)
  (h_new_diameter : new_diameter = 16) :
  let new_height := (initial_diameter^2 * initial_height) / new_diameter^2
  new_height = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_after_forging_l4068_406877


namespace NUMINAMATH_CALUDE_video_game_cost_l4068_406846

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def amount_left : ℕ := 41

def total_savings : ℕ := september_savings + october_savings + november_savings

theorem video_game_cost : total_savings - amount_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l4068_406846


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4068_406885

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) = s n + d

/-- The last term of a finite arithmetic sequence. -/
def last_term (s : ℕ → ℤ) (n : ℕ) : ℤ := s (n - 1)

theorem arithmetic_sequence_length :
  ∀ s : ℕ → ℤ,
  is_arithmetic_sequence s →
  s 0 = -3 →
  last_term s 13 = 45 →
  ∃ n : ℕ, n = 13 ∧ last_term s n = 45 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4068_406885


namespace NUMINAMATH_CALUDE_stellas_album_has_50_pages_l4068_406856

/-- Calculates the number of pages in Stella's stamp album --/
def stellas_album_pages (stamps_per_first_page : ℕ) (stamps_per_other_page : ℕ) (total_stamps : ℕ) : ℕ :=
  let first_pages := 10
  let stamps_in_first_pages := first_pages * stamps_per_first_page
  let remaining_stamps := total_stamps - stamps_in_first_pages
  let other_pages := remaining_stamps / stamps_per_other_page
  first_pages + other_pages

/-- Theorem stating that Stella's album has 50 pages --/
theorem stellas_album_has_50_pages :
  stellas_album_pages (5 * 30) 50 3500 = 50 := by
  sorry

#eval stellas_album_pages (5 * 30) 50 3500

end NUMINAMATH_CALUDE_stellas_album_has_50_pages_l4068_406856


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4068_406809

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The point (2,2) lies on the hyperbola -/
def point_on_hyperbola (h : Hyperbola) : Prop :=
  4 / h.a^2 - 4 / h.b^2 = 1

/-- The distance from the foci to the asymptotes equals the length of the real axis -/
def foci_distance_condition (h : Hyperbola) : Prop :=
  h.b = 2 * h.a

theorem hyperbola_equation (h : Hyperbola) 
  (h_point : point_on_hyperbola h) 
  (h_distance : foci_distance_condition h) : 
  h.a = Real.sqrt 3 ∧ h.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4068_406809


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l4068_406888

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F are on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l4068_406888


namespace NUMINAMATH_CALUDE_abs_neg_product_eq_product_l4068_406876

theorem abs_neg_product_eq_product {a b : ℝ} (ha : a < 0) (hb : 0 < b) : |-(a * b)| = a * b := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_product_eq_product_l4068_406876


namespace NUMINAMATH_CALUDE_lucas_100_mod_9_l4068_406841

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- Lucas sequence modulo 9 has a period of 12 -/
axiom lucas_mod_9_period (n : ℕ) : lucas (n + 12) % 9 = lucas n % 9

/-- The 4th term of Lucas sequence modulo 9 is 7 -/
axiom lucas_4_mod_9 : lucas 3 % 9 = 7

theorem lucas_100_mod_9 : lucas 99 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_100_mod_9_l4068_406841


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4068_406872

theorem simplify_and_evaluate (x : ℤ) 
  (h1 : -1 ≤ x ∧ x ≤ 1) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 1) : 
  ((((x^2 - 1) / (x^2 - 2*x + 1) + 1 / (1 - x)) : ℚ) / (x^2 : ℚ) * (x - 1)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4068_406872


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l4068_406826

/-- Check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Convert a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ k : ℕ, k > 30 → 
    (isPalindrome k 2 ∧ isPalindrome k 6) → 
    k ≥ 55 ∧ 
    isPalindrome 55 2 ∧ 
    isPalindrome 55 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l4068_406826


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l4068_406866

/-- Given squares A and B with perimeters 16 and 32 respectively, 
    when placed side by side to form square C, the perimeter of C is 48. -/
theorem square_perimeter_problem (A B C : ℝ → ℝ → Prop) :
  (∀ x, A x x → 4 * x = 16) →  -- Square A has perimeter 16
  (∀ y, B y y → 4 * y = 32) →  -- Square B has perimeter 32
  (∀ z, C z z → ∃ x y, A x x ∧ B y y ∧ z = x + y) →  -- C is formed by A and B side by side
  (∀ z, C z z → 4 * z = 48) :=  -- The perimeter of C is 48
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l4068_406866


namespace NUMINAMATH_CALUDE_count_is_2530_l4068_406811

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 10^4 satisfying s(11n) = 2s(n) -/
def count : ℕ := sorry

/-- Theorem stating the count is 2530 -/
theorem count_is_2530 : count = 2530 := by sorry

end NUMINAMATH_CALUDE_count_is_2530_l4068_406811


namespace NUMINAMATH_CALUDE_problem_solution_l4068_406810

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x + g x ≤ 7 ↔ -1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) ↔ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l4068_406810


namespace NUMINAMATH_CALUDE_borgnine_chimps_count_l4068_406827

/-- The number of chimps Borgnine has seen at the zoo -/
def num_chimps : ℕ := 25

/-- The total number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine needs to see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a lion, lizard, or chimp has -/
def legs_per_mammal_or_reptile : ℕ := 4

/-- The number of legs a tarantula has -/
def legs_per_tarantula : ℕ := 8

theorem borgnine_chimps_count :
  num_chimps * legs_per_mammal_or_reptile +
  num_lions * legs_per_mammal_or_reptile +
  num_lizards * legs_per_mammal_or_reptile +
  num_tarantulas * legs_per_tarantula = total_legs :=
by sorry

end NUMINAMATH_CALUDE_borgnine_chimps_count_l4068_406827


namespace NUMINAMATH_CALUDE_max_small_packages_with_nine_large_l4068_406882

/-- Represents the weight capacity of a service lift -/
structure LiftCapacity where
  large_packages : ℕ
  small_packages : ℕ

/-- Calculates the maximum number of small packages that can be carried alongside a given number of large packages -/
def max_small_packages (capacity : LiftCapacity) (large_count : ℕ) : ℕ :=
  let large_weight := capacity.small_packages / capacity.large_packages
  let remaining_capacity := capacity.small_packages - large_count * large_weight
  remaining_capacity

/-- Theorem: Given a lift with capacity of 12 large packages or 20 small packages,
    the maximum number of small packages that can be carried alongside 9 large packages is 5 -/
theorem max_small_packages_with_nine_large :
  let capacity := LiftCapacity.mk 12 20
  max_small_packages capacity 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_small_packages_with_nine_large_l4068_406882


namespace NUMINAMATH_CALUDE_fort_blocks_theorem_l4068_406873

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - wallThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  outerVolume - innerVolume

/-- Theorem stating that a fort with given dimensions requires 480 blocks -/
theorem fort_blocks_theorem :
  let fortDimensions : FortDimensions := ⟨15, 8, 6⟩
  let wallThickness : ℕ := 3/2
  blocksNeeded fortDimensions wallThickness = 480 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_theorem_l4068_406873


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4068_406824

/-- Given a hyperbola with eccentricity √5 and one vertex at (1, 0), 
    prove that its equation is x^2 - y^2/4 = 1 -/
theorem hyperbola_equation (e : ℝ) (v : ℝ × ℝ) :
  e = Real.sqrt 5 →
  v = (1, 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ↔ x^2 - y^2/4 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4068_406824


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l4068_406807

/-- The number of ways to arrange students from 5 grades visiting 5 museums,
    with exactly 2 grades visiting the Jia Science Museum -/
def arrangement_count : ℕ :=
  Nat.choose 5 2 * (4^3)

/-- Theorem stating that the number of arrangements is correct -/
theorem arrangement_count_correct :
  arrangement_count = Nat.choose 5 2 * (4^3) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l4068_406807


namespace NUMINAMATH_CALUDE_solution_product_l4068_406840

theorem solution_product (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 55 ∧ 
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 55 →
  (r + 4) * (s + 4) = 25 := by sorry

end NUMINAMATH_CALUDE_solution_product_l4068_406840


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l4068_406852

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_2_pow_20_minus_1 :
  unitsDigit ((2 ^ 20) - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l4068_406852


namespace NUMINAMATH_CALUDE_no_odd_prime_sum_107_l4068_406834

theorem no_odd_prime_sum_107 : ¬∃ (p q k : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd p ∧ 
  Odd q ∧ 
  p + q = 107 ∧ 
  p * q = k :=
sorry

end NUMINAMATH_CALUDE_no_odd_prime_sum_107_l4068_406834


namespace NUMINAMATH_CALUDE_separating_chord_length_l4068_406850

/-- Represents a hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- The lengths of the sides
  side_lengths : Fin 6 → ℝ
  -- Condition that alternating sides have lengths 5 and 4
  alternating_sides : ∀ i : Fin 6, side_lengths i = if i % 2 = 0 then 5 else 4

/-- The chord that separates the hexagon into two trapezoids -/
def separating_chord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the separating chord -/
theorem separating_chord_length (h : InscribedHexagon) :
  separating_chord h = 180 / 49 := by sorry

end NUMINAMATH_CALUDE_separating_chord_length_l4068_406850


namespace NUMINAMATH_CALUDE_square_sum_problem_l4068_406806

theorem square_sum_problem (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁^2 + 5*x₂^2 = 10)
  (h2 : x₂*y₁ - x₁*y₂ = 5)
  (h3 : x₁*y₁ + 5*x₂*y₂ = Real.sqrt 105) :
  y₁^2 + 5*y₂^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_problem_l4068_406806
