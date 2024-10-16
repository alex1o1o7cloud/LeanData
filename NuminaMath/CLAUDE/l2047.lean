import Mathlib

namespace NUMINAMATH_CALUDE_airplane_purchase_exceeds_budget_l2047_204705

/-- Proves that the total cost of purchasing the airplane exceeds $5.00 USD -/
theorem airplane_purchase_exceeds_budget : 
  let initial_budget : ℝ := 5.00
  let airplane_cost_eur : ℝ := 3.80
  let exchange_rate : ℝ := 0.82
  let sales_tax_rate : ℝ := 0.075
  let credit_card_surcharge_rate : ℝ := 0.035
  let processing_fee_usd : ℝ := 0.25
  
  let airplane_cost_usd : ℝ := airplane_cost_eur / exchange_rate
  let sales_tax : ℝ := airplane_cost_usd * sales_tax_rate
  let credit_card_surcharge : ℝ := airplane_cost_usd * credit_card_surcharge_rate
  let total_cost : ℝ := airplane_cost_usd + sales_tax + credit_card_surcharge + processing_fee_usd
  
  total_cost > initial_budget := by
  sorry

#check airplane_purchase_exceeds_budget

end NUMINAMATH_CALUDE_airplane_purchase_exceeds_budget_l2047_204705


namespace NUMINAMATH_CALUDE_exists_crocodile_coloring_l2047_204723

/-- A coloring function for the infinite chess grid -/
def GridColoring := ℤ → ℤ → Fin 2

/-- The crocodile move property for a given coloring -/
def IsCrocodileColoring (f : GridColoring) (m n : ℕ+) : Prop :=
  ∀ x y : ℤ, f x y ≠ f (x + m) (y + n) ∧ f x y ≠ f (x + n) (y + m)

/-- Theorem: For any positive integers m and n, there exists a valid crocodile coloring -/
theorem exists_crocodile_coloring (m n : ℕ+) :
  ∃ f : GridColoring, IsCrocodileColoring f m n := by
  sorry

end NUMINAMATH_CALUDE_exists_crocodile_coloring_l2047_204723


namespace NUMINAMATH_CALUDE_triangle_area_l2047_204787

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : θ = 2 * Real.pi / 3) :
  let area := (1/2) * a * b * Real.sin θ
  area = (3 * Real.sqrt 3) / 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2047_204787


namespace NUMINAMATH_CALUDE_duck_pricing_problem_l2047_204710

/-- A problem about duck pricing and profit -/
theorem duck_pricing_problem 
  (num_ducks : ℕ) 
  (weight_per_duck : ℝ) 
  (selling_price_per_pound : ℝ) 
  (total_profit : ℝ) 
  (h1 : num_ducks = 30)
  (h2 : weight_per_duck = 4)
  (h3 : selling_price_per_pound = 5)
  (h4 : total_profit = 300) :
  let total_revenue := num_ducks * weight_per_duck * selling_price_per_pound
  let price_per_duck := (total_revenue - total_profit) / num_ducks
  price_per_duck = 10 := by
sorry

end NUMINAMATH_CALUDE_duck_pricing_problem_l2047_204710


namespace NUMINAMATH_CALUDE_max_green_lily_students_l2047_204779

-- Define variables
variable (x : ℝ) -- Cost of green lily
variable (y : ℝ) -- Cost of spider plant
variable (m : ℝ) -- Number of students taking care of green lilies

-- Define conditions
axiom condition1 : 2 * x + 3 * y = 36
axiom condition2 : x + 2 * y = 21
axiom total_students : m + (48 - m) = 48
axiom cost_constraint : m * x + (48 - m) * y ≤ 378

-- Theorem to prove
theorem max_green_lily_students : 
  ∃ m : ℝ, m ≤ 30 ∧ 
  ∀ n : ℝ, (n * x + (48 - n) * y ≤ 378 → n ≤ m) :=
sorry

end NUMINAMATH_CALUDE_max_green_lily_students_l2047_204779


namespace NUMINAMATH_CALUDE_elevation_change_proof_l2047_204777

def initial_elevation : ℝ := 400

def stage1_rate : ℝ := 10
def stage1_time : ℝ := 5

def stage2_rate : ℝ := 15
def stage2_time : ℝ := 3

def stage3_rate : ℝ := 12
def stage3_time : ℝ := 6

def stage4_rate : ℝ := 8
def stage4_time : ℝ := 4

def stage5_rate : ℝ := 5
def stage5_time : ℝ := 2

def final_elevation : ℝ := initial_elevation - 
  (stage1_rate * stage1_time + 
   stage2_rate * stage2_time + 
   stage3_rate * stage3_time - 
   stage4_rate * stage4_time + 
   stage5_rate * stage5_time)

theorem elevation_change_proof : final_elevation = 255 := by sorry

end NUMINAMATH_CALUDE_elevation_change_proof_l2047_204777


namespace NUMINAMATH_CALUDE_problem_statement_l2047_204768

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2047_204768


namespace NUMINAMATH_CALUDE_compute_expression_l2047_204743

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2047_204743


namespace NUMINAMATH_CALUDE_triangle_side_length_l2047_204707

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, a + b = 6, and (a*cos B + b*cos A) / c = 2*cos C, then c = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  a + b = 6 →
  (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C →
  c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2047_204707


namespace NUMINAMATH_CALUDE_admission_price_is_two_l2047_204783

/-- Calculates the admission price for adults given the total number of people,
    admission price for children, total admission receipts, and number of adults. -/
def admission_price_for_adults (total_people : ℕ) (child_price : ℚ) 
                               (total_receipts : ℚ) (num_adults : ℕ) : ℚ :=
  (total_receipts - (total_people - num_adults : ℚ) * child_price) / num_adults

/-- Proves that the admission price for adults is $2 given the specific conditions. -/
theorem admission_price_is_two :
  admission_price_for_adults 610 1 960 350 = 2 := by sorry

end NUMINAMATH_CALUDE_admission_price_is_two_l2047_204783


namespace NUMINAMATH_CALUDE_eleventh_term_is_320_l2047_204717

/-- A geometric sequence with given 5th and 8th terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  fifth_term : a 5 = 5
  eighth_term : a 8 = 40

/-- The 11th term of the geometric sequence is 320 -/
theorem eleventh_term_is_320 (seq : GeometricSequence) : seq.a 11 = 320 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_320_l2047_204717


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2047_204793

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k + 1) * x - 6
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (f 2 = 0 → k = -2 ∧ f (-3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2047_204793


namespace NUMINAMATH_CALUDE_blue_tshirts_per_pack_l2047_204751

/-- Given the following:
  * Dave bought 3 packs of white T-shirts and 2 packs of blue T-shirts
  * White T-shirts come in packs of 6
  * Dave bought 26 T-shirts in total
Prove that the number of blue T-shirts in each pack is 4 -/
theorem blue_tshirts_per_pack (white_packs : ℕ) (blue_packs : ℕ) (white_per_pack : ℕ) (total_tshirts : ℕ)
  (h1 : white_packs = 3)
  (h2 : blue_packs = 2)
  (h3 : white_per_pack = 6)
  (h4 : total_tshirts = 26)
  (h5 : white_packs * white_per_pack + blue_packs * (total_tshirts - white_packs * white_per_pack) / blue_packs = total_tshirts) :
  (total_tshirts - white_packs * white_per_pack) / blue_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_tshirts_per_pack_l2047_204751


namespace NUMINAMATH_CALUDE_girls_in_school_l2047_204791

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_boys_diff : ℕ) :
  total_students = 1750 →
  sample_size = 250 →
  girls_boys_diff = 20 →
  ∃ (girls_in_school : ℕ),
    girls_in_school = 805 ∧
    girls_in_school * sample_size = (sample_size - girls_boys_diff) * total_students / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l2047_204791


namespace NUMINAMATH_CALUDE_cookie_count_equivalence_l2047_204781

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Circle
  | Rectangle
  | Parallelogram
  | Triangle
  | Square

/-- Represents a friend who bakes cookies -/
structure Friend where
  name : String
  shape : CookieShape

/-- Represents the dimensions of a cookie -/
structure CookieDimensions where
  base : ℝ
  height : ℝ

theorem cookie_count_equivalence 
  (friends : List Friend)
  (carlos_dims : CookieDimensions)
  (lisa_side : ℝ)
  (carlos_count : ℕ)
  (h1 : friends.length = 5)
  (h2 : ∃ f ∈ friends, f.name = "Carlos" ∧ f.shape = CookieShape.Triangle)
  (h3 : ∃ f ∈ friends, f.name = "Lisa" ∧ f.shape = CookieShape.Square)
  (h4 : carlos_dims.base = 4)
  (h5 : carlos_dims.height = 5)
  (h6 : carlos_count = 20)
  (h7 : lisa_side = 5)
  : (200 : ℝ) / (lisa_side ^ 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_equivalence_l2047_204781


namespace NUMINAMATH_CALUDE_power_of_power_l2047_204720

theorem power_of_power (a : ℝ) : (a^2)^4 = a^8 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2047_204720


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2047_204716

theorem polynomial_simplification (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2047_204716


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2047_204795

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25) 
  (h2 : failed_english = 35) 
  (h3 : failed_both = 40) : 
  100 - (failed_hindi + failed_english - failed_both) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2047_204795


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2047_204754

def M : Set Int := {-1, 1}
def N : Set Int := {-2, 1, 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2047_204754


namespace NUMINAMATH_CALUDE_union_A_complement_B_A_subset_B_iff_a_in_range_l2047_204753

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | 2*x^2 - 3*x - 2 < 0}

-- Part 1
theorem union_A_complement_B :
  A 1 ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Part 2
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_A_subset_B_iff_a_in_range_l2047_204753


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l2047_204750

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12

theorem total_weight_of_balls :
  blue_ball_weight + brown_ball_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l2047_204750


namespace NUMINAMATH_CALUDE_gcd_40304_30203_l2047_204747

theorem gcd_40304_30203 : Nat.gcd 40304 30203 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_40304_30203_l2047_204747


namespace NUMINAMATH_CALUDE_total_questions_to_review_l2047_204715

-- Define the given conditions
def num_classes : ℕ := 5
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10

-- State the theorem
theorem total_questions_to_review :
  num_classes * students_per_class * questions_per_exam = 1750 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_to_review_l2047_204715


namespace NUMINAMATH_CALUDE_D_72_l2047_204713

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where order matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) = 43 -/
theorem D_72 : D 72 = 43 := by sorry

end NUMINAMATH_CALUDE_D_72_l2047_204713


namespace NUMINAMATH_CALUDE_border_area_is_198_l2047_204733

/-- Calculates the area of the border for a framed picture -/
def border_area (picture_height : ℕ) (picture_width : ℕ) (border_width : ℕ) : ℕ :=
  let total_height := picture_height + 2 * border_width
  let total_width := picture_width + 2 * border_width
  total_height * total_width - picture_height * picture_width

/-- Theorem stating that the border area for the given dimensions is 198 square inches -/
theorem border_area_is_198 :
  border_area 12 15 3 = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l2047_204733


namespace NUMINAMATH_CALUDE_black_socks_bought_l2047_204737

/-- Represents the number of pairs of socks of each color -/
structure SockCount where
  blue : ℕ
  black : ℕ
  white : ℕ

/-- The initial sock count before buying more black socks -/
def initialSocks : SockCount :=
  { blue := 6, black := 18, white := 12 }

/-- The proportion of black socks after buying more -/
def blackProportion : ℚ := 3 / 5

/-- Calculates the total number of sock pairs -/
def totalSocks (s : SockCount) : ℕ :=
  s.blue + s.black + s.white

/-- Theorem stating the number of black sock pairs Dmitry bought -/
theorem black_socks_bought (x : ℕ) : 
  (initialSocks.black + x : ℚ) / (totalSocks initialSocks + x : ℚ) = blackProportion →
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_black_socks_bought_l2047_204737


namespace NUMINAMATH_CALUDE_cubes_equation_solution_l2047_204758

theorem cubes_equation_solution (x y z : ℤ) (h : x^3 + 2*y^3 = 4*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubes_equation_solution_l2047_204758


namespace NUMINAMATH_CALUDE_line_contains_point_l2047_204736

/-- Given a line with equation -2/3 - 3kx = 7y that contains the point (1/3, -5), 
    prove that the value of k is 103/3. -/
theorem line_contains_point (k : ℚ) : 
  (-2/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-5 : ℚ) → k = 103/3 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l2047_204736


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2047_204726

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2047_204726


namespace NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2047_204732

theorem triangle_angle_b_is_pi_third (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2  -- Given condition
  → B = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2047_204732


namespace NUMINAMATH_CALUDE_mean_of_points_l2047_204740

def points : List ℝ := [81, 73, 83, 86, 73]

theorem mean_of_points : (points.sum / points.length : ℝ) = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_points_l2047_204740


namespace NUMINAMATH_CALUDE_part_one_part_two_l2047_204764

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 ≤ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part I
theorem part_one (m : ℝ) : A m ∩ B = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) : B ⊆ (Set.univ \ A m) → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2047_204764


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2047_204738

/-- Given n ≥ 2 distinct integers, the polynomial f(x) = (x - a₁)(x - a₂) ... (x - aₙ) - 1 is irreducible over the integers. -/
theorem polynomial_irreducibility (n : ℕ) (a : Fin n → ℤ) (h1 : n ≥ 2) (h2 : Function.Injective a) :
  Irreducible (((Polynomial.X : Polynomial ℤ) - (Finset.univ.prod (fun i => Polynomial.X - Polynomial.C (a i)))) - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2047_204738


namespace NUMINAMATH_CALUDE_original_total_is_390_l2047_204786

/-- Represents the number of movies in each format --/
structure MovieCollection where
  dvd : ℕ
  bluray : ℕ
  digital : ℕ

/-- The original collection of movies --/
def original : MovieCollection := sorry

/-- The updated collection after purchasing new movies --/
def updated : MovieCollection := sorry

/-- The ratio of the original collection --/
def original_ratio : MovieCollection := ⟨7, 2, 1⟩

/-- The ratio of the updated collection --/
def updated_ratio : MovieCollection := ⟨13, 4, 2⟩

/-- The number of new Blu-ray movies purchased --/
def new_bluray : ℕ := 5

/-- The number of new digital movies purchased --/
def new_digital : ℕ := 3

theorem original_total_is_390 :
  ∃ (x : ℕ),
    original.dvd = 7 * x ∧
    original.bluray = 2 * x ∧
    original.digital = x ∧
    updated.dvd = original.dvd ∧
    updated.bluray = original.bluray + new_bluray ∧
    updated.digital = original.digital + new_digital ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.bluray : ℚ) / updated_ratio.bluray ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.digital : ℚ) / updated_ratio.digital ∧
    original.dvd + original.bluray + original.digital = 390 :=
by
  sorry

end NUMINAMATH_CALUDE_original_total_is_390_l2047_204786


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l2047_204704

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Given the endpoints of the major axis and a point on the ellipse, 
    calculate the ellipse parameters -/
def calculateEllipse (p1 p2 p3 : Point) : Ellipse :=
  sorry

/-- Calculate the area of an ellipse -/
def ellipseArea (e : Ellipse) : ℝ :=
  sorry

/-- The main theorem stating the area of the specific ellipse -/
theorem specific_ellipse_area : 
  let p1 : Point := ⟨-10, 3⟩
  let p2 : Point := ⟨8, 3⟩
  let p3 : Point := ⟨6, 8⟩
  let e : Ellipse := calculateEllipse p1 p2 p3
  ellipseArea e = (405 * Real.pi) / (4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l2047_204704


namespace NUMINAMATH_CALUDE_solve_for_x_l2047_204709

theorem solve_for_x (x y : ℚ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2047_204709


namespace NUMINAMATH_CALUDE_union_equals_A_l2047_204702

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {y | m*y + 2 = 0}

theorem union_equals_A : {m : ℝ | A ∪ B m = A} = {0, -1, -2/3} := by sorry

end NUMINAMATH_CALUDE_union_equals_A_l2047_204702


namespace NUMINAMATH_CALUDE_package_cost_proof_l2047_204761

/-- The cost of a 12-roll package of paper towels -/
def package_cost : ℝ := 9

/-- The cost of one roll sold individually -/
def individual_roll_cost : ℝ := 1

/-- The number of rolls in a package -/
def rolls_per_package : ℕ := 12

/-- The percent of savings per roll for the package -/
def savings_percent : ℝ := 25

theorem package_cost_proof : 
  package_cost = rolls_per_package * (individual_roll_cost * (1 - savings_percent / 100)) :=
by sorry

end NUMINAMATH_CALUDE_package_cost_proof_l2047_204761


namespace NUMINAMATH_CALUDE_drops_used_proof_l2047_204792

/-- Represents the number of drops used to test a single beaker -/
def drops_per_beaker : ℕ := 3

/-- Represents the total number of beakers -/
def total_beakers : ℕ := 22

/-- Represents the number of beakers with copper ions -/
def copper_beakers : ℕ := 8

/-- Represents the number of beakers without copper ions that were tested -/
def tested_non_copper : ℕ := 7

theorem drops_used_proof :
  drops_per_beaker * (copper_beakers + tested_non_copper) = 45 := by
  sorry

end NUMINAMATH_CALUDE_drops_used_proof_l2047_204792


namespace NUMINAMATH_CALUDE_exists_player_in_interval_l2047_204730

/-- Represents a round-robin tennis tournament -/
structure Tournament :=
  (n : ℕ)  -- Half the number of players minus 1
  (k : ℕ)  -- Number of matches won by the weaker player

/-- The number of wins for each player -/
def wins (t : Tournament) : Fin (2 * t.n + 1) → ℕ :=
  sorry

/-- Theorem stating the existence of a player with wins in the specified interval -/
theorem exists_player_in_interval (t : Tournament) :
  ∃ (p : Fin (2 * t.n + 1)),
    (t.n : ℝ) - Real.sqrt (2 * t.k) ≤ (wins t p) ∧
    (wins t p : ℝ) ≤ t.n + Real.sqrt (2 * t.k) :=
  sorry

end NUMINAMATH_CALUDE_exists_player_in_interval_l2047_204730


namespace NUMINAMATH_CALUDE_mom_initial_money_l2047_204722

/-- The amount of money Mom spent on bananas -/
def banana_cost : ℕ := 2 * 4

/-- The amount of money Mom spent on pears -/
def pear_cost : ℕ := 2

/-- The amount of money Mom spent on asparagus -/
def asparagus_cost : ℕ := 6

/-- The amount of money Mom spent on chicken -/
def chicken_cost : ℕ := 11

/-- The amount of money Mom has left after shopping -/
def money_left : ℕ := 28

/-- The total amount Mom spent on groceries -/
def total_spent : ℕ := banana_cost + pear_cost + asparagus_cost + chicken_cost

/-- Theorem stating that Mom had €55 when she left for the market -/
theorem mom_initial_money : total_spent + money_left = 55 := by
  sorry

end NUMINAMATH_CALUDE_mom_initial_money_l2047_204722


namespace NUMINAMATH_CALUDE_max_min_product_l2047_204774

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 15) (hprod : x*y + y*z + z*x = 45) :
  ∃ (m : ℝ), m = min (x*y) (min (y*z) (z*x)) ∧ m ≤ 17.5 ∧
  ∀ (m' : ℝ), m' = min (x*y) (min (y*z) (z*x)) → m' ≤ 17.5 := by
sorry

end NUMINAMATH_CALUDE_max_min_product_l2047_204774


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2047_204728

theorem ten_thousandths_digit_of_seven_thirty_seconds (f : ℚ) (d : ℕ) : 
  f = 7 / 32 →
  d = (⌊f * 10000⌋ % 10) →
  d = 8 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2047_204728


namespace NUMINAMATH_CALUDE_factor_expression_l2047_204788

theorem factor_expression : ∀ x : ℝ, 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2047_204788


namespace NUMINAMATH_CALUDE_f_has_two_extreme_points_l2047_204745

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3 - 9

-- Define what an extreme point is
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≠ f x

-- State the theorem
theorem f_has_two_extreme_points :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, is_extreme_point f x :=
sorry

end NUMINAMATH_CALUDE_f_has_two_extreme_points_l2047_204745


namespace NUMINAMATH_CALUDE_five_hour_charge_l2047_204799

/-- Represents the charge structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  moreExpensiveFirst : firstHourCharge = additionalHourCharge + 35
  twoHourTotal : firstHourCharge + additionalHourCharge = 161

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating that the total charge for 5 hours of therapy is $350. -/
theorem five_hour_charge (charges : TherapyCharges) : totalCharge charges 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_charge_l2047_204799


namespace NUMINAMATH_CALUDE_cookie_distribution_l2047_204701

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  total_cookies = num_people * cookies_per_person →
  cookies_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2047_204701


namespace NUMINAMATH_CALUDE_division_problem_l2047_204748

theorem division_problem (A : ℕ) (h1 : 59 / A = 6) (h2 : 59 % A = 5) : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2047_204748


namespace NUMINAMATH_CALUDE_combination_permutation_ratio_l2047_204789

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem combination_permutation_ratio (x y : ℕ) (h : y > x) :
  (binomial_coefficient y x : ℚ) / (binomial_coefficient (y + 2) x : ℚ) = 1 / 3 ∧
  (permutation y x : ℚ) / (binomial_coefficient y x : ℚ) = 24 ↔
  x = 4 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_ratio_l2047_204789


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2047_204775

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes with no empty boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2047_204775


namespace NUMINAMATH_CALUDE_angle_bisector_exists_l2047_204708

/-- A ruler with constant width and parallel edges -/
structure ConstantWidthRuler where
  width : ℝ
  width_positive : width > 0

/-- An angle in a plane -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line bisects an angle -/
def bisects (l : Line) (a : Angle) : Prop :=
  sorry

/-- Predicate to check if a line can be constructed using a constant width ruler -/
def constructible_with_ruler (l : Line) (r : ConstantWidthRuler) : Prop :=
  sorry

/-- Theorem stating that for any angle, there exists a bisector constructible with a constant width ruler -/
theorem angle_bisector_exists (a : Angle) (r : ConstantWidthRuler) :
  ∃ l : Line, bisects l a ∧ constructible_with_ruler l r := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_exists_l2047_204708


namespace NUMINAMATH_CALUDE_total_pears_picked_l2047_204739

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l2047_204739


namespace NUMINAMATH_CALUDE_four_fours_l2047_204778

def four_digit_expr : ℕ → Prop :=
  fun n => ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ),
    (e 4 4 4 4 = n) ∧
    (∀ x y z w, e x y z w = n → x = 4 ∧ y = 4 ∧ z = 4 ∧ w = 4)

theorem four_fours :
  four_digit_expr 3 ∧
  four_digit_expr 4 ∧
  four_digit_expr 5 ∧
  four_digit_expr 6 := by sorry

end NUMINAMATH_CALUDE_four_fours_l2047_204778


namespace NUMINAMATH_CALUDE_compare_negative_mixed_numbers_l2047_204771

theorem compare_negative_mixed_numbers :
  -6.5 > -(6 + 3/5) := by sorry

end NUMINAMATH_CALUDE_compare_negative_mixed_numbers_l2047_204771


namespace NUMINAMATH_CALUDE_interest_gender_association_prob_not_interested_given_boy_expected_value_X_l2047_204741

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of boys
def num_boys : ℕ := 55

-- Define the number of interested boys
def interested_boys : ℕ := 45

-- Define the number of interested girls
def interested_girls : ℕ := 20

-- Define the significance level
def alpha : ℚ := 1/200

-- Define the critical value for the given alpha
def critical_value : ℚ := 7879/1000

-- Function to calculate chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem stating the association between interest and gender
theorem interest_gender_association :
  let a := interested_boys
  let b := num_boys - interested_boys
  let c := interested_girls
  let d := total_students - num_boys - interested_girls
  chi_square a b c d > critical_value := by sorry

-- Function to calculate conditional probability
def conditional_probability (event_ab event_a : ℕ) : ℚ :=
  event_ab / event_a

-- Theorem for the conditional probability P(B|A)
theorem prob_not_interested_given_boy :
  conditional_probability (num_boys - interested_boys) num_boys = 2/11 := by sorry

-- Function to calculate expected value
def expected_value (p₀ p₁ p₂ p₃ : ℚ) : ℚ :=
  0 * p₀ + 1 * p₁ + 2 * p₂ + 3 * p₃

-- Theorem for the expected value of X
theorem expected_value_X :
  expected_value (4/35) (18/35) (12/35) (1/35) = 9/7 := by sorry

end NUMINAMATH_CALUDE_interest_gender_association_prob_not_interested_given_boy_expected_value_X_l2047_204741


namespace NUMINAMATH_CALUDE_last_three_positions_l2047_204760

/-- Represents the position of a person in the line after a certain number of rounds -/
def position (round : ℕ) : ℕ :=
  match round with
  | 0 => 3
  | n + 1 =>
    let prev := position n
    if prev % 2 = 1 then (3 * prev - 1) / 2 else (3 * prev - 2) / 2

/-- The theorem stating the initial positions of the last three people remaining -/
theorem last_three_positions (initial_count : ℕ) (h : initial_count = 2009) :
  ∃ (rounds : ℕ), position rounds = 1600 ∧ 
    (∀ k, k > rounds → position k < 1600) ∧
    (∀ n, n ≤ initial_count → n ≠ 1 → n ≠ 2 → n ≠ 1600 → 
      ∃ m, m ≤ rounds ∧ (3 * (position m)) % n = 0) :=
sorry

end NUMINAMATH_CALUDE_last_three_positions_l2047_204760


namespace NUMINAMATH_CALUDE_white_beans_count_l2047_204749

/-- The number of white jelly beans in one bag -/
def white_beans_in_bag : ℕ := sorry

/-- The number of bags needed to fill the fishbowl -/
def bags_in_fishbowl : ℕ := 3

/-- The number of red jelly beans in one bag -/
def red_beans_in_bag : ℕ := 24

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_fishbowl : ℕ := 126

theorem white_beans_count : white_beans_in_bag = 18 := by
  sorry

end NUMINAMATH_CALUDE_white_beans_count_l2047_204749


namespace NUMINAMATH_CALUDE_project_completion_theorem_l2047_204719

/-- The number of days to complete a project given two workers with different rates -/
def project_completion_days (a_rate b_rate : ℚ) (a_quit_before : ℕ) : ℕ :=
  let total_days := 20
  total_days

theorem project_completion_theorem (a_rate b_rate : ℚ) (a_quit_before : ℕ) :
  a_rate = 1/20 ∧ b_rate = 1/40 ∧ a_quit_before = 10 →
  (project_completion_days a_rate b_rate a_quit_before - a_quit_before) * a_rate +
  project_completion_days a_rate b_rate a_quit_before * b_rate = 1 :=
by
  sorry

#eval project_completion_days (1/20) (1/40) 10

end NUMINAMATH_CALUDE_project_completion_theorem_l2047_204719


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l2047_204735

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l2047_204735


namespace NUMINAMATH_CALUDE_alexis_bought_21_pants_l2047_204772

/-- Given information about Isabella and Alexis's shopping -/
structure ShoppingInfo where
  isabella_total : ℕ
  alexis_dresses : ℕ
  alexis_multiplier : ℕ

/-- Calculates the number of pants Alexis bought -/
def alexis_pants (info : ShoppingInfo) : ℕ :=
  info.alexis_multiplier * (info.isabella_total - (info.alexis_dresses / info.alexis_multiplier))

/-- Theorem stating that Alexis bought 21 pants given the shopping information -/
theorem alexis_bought_21_pants (info : ShoppingInfo) 
  (h1 : info.isabella_total = 13)
  (h2 : info.alexis_dresses = 18)
  (h3 : info.alexis_multiplier = 3) : 
  alexis_pants info = 21 := by
  sorry

#eval alexis_pants ⟨13, 18, 3⟩

end NUMINAMATH_CALUDE_alexis_bought_21_pants_l2047_204772


namespace NUMINAMATH_CALUDE_triangular_pyramid_volume_l2047_204711

-- Define a triangular pyramid with edge length √2
def TriangularPyramid := {edge_length : ℝ // edge_length = Real.sqrt 2}

-- Define the volume of a triangular pyramid
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  -- The actual calculation of the volume is not implemented here
  -- We're just declaring that such a function exists
  sorry

-- Theorem statement
theorem triangular_pyramid_volume (p : TriangularPyramid) : volume p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_volume_l2047_204711


namespace NUMINAMATH_CALUDE_normal_wage_is_twelve_l2047_204714

/-- Calculates the total earnings for a worker given their normal hourly wage,
    total hours worked, and an overtime multiplier for hours over 40. -/
def totalEarnings (normalWage : ℚ) (hoursWorked : ℕ) (overtimeMultiplier : ℚ) : ℚ :=
  if hoursWorked ≤ 40 then
    normalWage * hoursWorked
  else
    normalWage * 40 + normalWage * overtimeMultiplier * (hoursWorked - 40)

/-- Proves that given the specified conditions, the worker's normal hourly wage is $12. -/
theorem normal_wage_is_twelve :
  ∃ (normalWage : ℚ),
    normalWage > 0 ∧
    totalEarnings normalWage 52 (3/2) = 696 ∧
    normalWage = 12 := by
  sorry

end NUMINAMATH_CALUDE_normal_wage_is_twelve_l2047_204714


namespace NUMINAMATH_CALUDE_simplify_expression_l2047_204773

theorem simplify_expression : 18 * (7/8) * (1/12)^2 = 7/768 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2047_204773


namespace NUMINAMATH_CALUDE_inverse_cube_relation_l2047_204798

/-- Given that y varies inversely as the cube of x, prove that x = ∛3 when y = 18, 
    given that y = 2 when x = 3. -/
theorem inverse_cube_relation (x y : ℝ) (k : ℝ) (h1 : y * x^3 = k) 
  (h2 : 2 * 3^3 = k) (h3 : 18 * x^3 = k) : x = (3 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_cube_relation_l2047_204798


namespace NUMINAMATH_CALUDE_unique_prime_plus_10_14_prime_l2047_204746

theorem unique_prime_plus_10_14_prime :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_plus_10_14_prime_l2047_204746


namespace NUMINAMATH_CALUDE_sum_of_coordinates_X_l2047_204756

def Y : ℝ × ℝ := (2, 9)
def Z : ℝ × ℝ := (1, 5)

theorem sum_of_coordinates_X (X : ℝ × ℝ) 
  (h1 : (dist X Z) / (dist X Y) = 3 / 4)
  (h2 : (dist Z Y) / (dist X Y) = 1 / 4) : 
  X.1 + X.2 = -9 := by
  sorry

#check sum_of_coordinates_X

end NUMINAMATH_CALUDE_sum_of_coordinates_X_l2047_204756


namespace NUMINAMATH_CALUDE_percentage_men_science_majors_l2047_204721

/-- Represents the composition of a college class -/
structure ClassComposition where
  total : ℝ
  women : ℝ
  men : ℝ
  scienceMajors : ℝ
  womenScienceMajors : ℝ
  nonScienceMajors : ℝ

/-- Theorem stating the percentage of men who are science majors -/
theorem percentage_men_science_majors (c : ClassComposition) : 
  c.total > 0 ∧ 
  c.women = 0.6 * c.total ∧ 
  c.men = 0.4 * c.total ∧ 
  c.nonScienceMajors = 0.6 * c.total ∧
  c.womenScienceMajors = 0.2 * c.women →
  (c.scienceMajors - c.womenScienceMajors) / c.men = 0.7 := by
  sorry

#check percentage_men_science_majors

end NUMINAMATH_CALUDE_percentage_men_science_majors_l2047_204721


namespace NUMINAMATH_CALUDE_integer_root_quadratic_count_l2047_204767

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), 
    Finset.card S = 8 ∧ 
    (∀ a ∈ S, ∃ r s : ℤ, 
      (∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = r ∨ x = s)) :=
sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_count_l2047_204767


namespace NUMINAMATH_CALUDE_equal_sequence_l2047_204755

theorem equal_sequence (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv (Fin 2011) (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
by sorry

end NUMINAMATH_CALUDE_equal_sequence_l2047_204755


namespace NUMINAMATH_CALUDE_hydroflow_pump_calculation_l2047_204780

/-- The rate at which the Hydroflow system pumps water, in gallons per hour -/
def pump_rate : ℝ := 360

/-- The time in minutes for which we want to calculate the amount of water pumped -/
def pump_time : ℝ := 30

/-- Theorem stating that the Hydroflow system pumps 180 gallons in 30 minutes -/
theorem hydroflow_pump_calculation : 
  pump_rate * (pump_time / 60) = 180 := by sorry

end NUMINAMATH_CALUDE_hydroflow_pump_calculation_l2047_204780


namespace NUMINAMATH_CALUDE_value_of_difference_product_l2047_204744

theorem value_of_difference_product (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by
  sorry

end NUMINAMATH_CALUDE_value_of_difference_product_l2047_204744


namespace NUMINAMATH_CALUDE_emily_age_is_23_l2047_204734

-- Define the ages as natural numbers
def uncle_bob_age : ℕ := 54
def daniel_age : ℕ := uncle_bob_age / 2
def emily_age : ℕ := daniel_age - 4
def zoe_age : ℕ := emily_age * 3 / 2

-- Theorem statement
theorem emily_age_is_23 : emily_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_emily_age_is_23_l2047_204734


namespace NUMINAMATH_CALUDE_tensor_inequality_implies_a_bound_l2047_204706

-- Define the ⊗ operation
def tensor (x y : ℝ) := x * (1 - y)

-- Define the main theorem
theorem tensor_inequality_implies_a_bound (a : ℝ) : 
  (∀ x > 2, tensor (x - a) x ≤ a + 2) → a ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_tensor_inequality_implies_a_bound_l2047_204706


namespace NUMINAMATH_CALUDE_trig_identity_l2047_204770

theorem trig_identity (α : Real) (h : Real.tan α = 2) : 
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2047_204770


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l2047_204797

/-- The number of factors of 1200 that are perfect squares -/
def perfect_square_factors : ℕ :=
  let n := 1200
  let prime_factorization := (2, 4) :: (3, 1) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of factors of 1200 that are perfect squares is 6 -/
theorem count_perfect_square_factors :
  perfect_square_factors = 6 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l2047_204797


namespace NUMINAMATH_CALUDE_f_properties_l2047_204790

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (Set.range f = Set.Ici (-1 / (2 * Real.exp 1))) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2047_204790


namespace NUMINAMATH_CALUDE_beginner_course_fraction_l2047_204766

theorem beginner_course_fraction :
  ∀ (total_students : ℕ) (calculus_students : ℕ) (trigonometry_students : ℕ) 
    (beginner_calculus : ℕ) (beginner_trigonometry : ℕ),
  total_students > 0 →
  calculus_students + trigonometry_students = total_students →
  trigonometry_students = (3 * calculus_students) / 2 →
  beginner_calculus = (4 * calculus_students) / 5 →
  (beginner_trigonometry : ℚ) / total_students = 48 / 100 →
  (beginner_calculus + beginner_trigonometry : ℚ) / total_students = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_beginner_course_fraction_l2047_204766


namespace NUMINAMATH_CALUDE_sundae_price_l2047_204794

theorem sundae_price 
  (ice_cream_bars : ℕ) 
  (sundaes : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) :
  ice_cream_bars = 225 →
  sundaes = 125 →
  total_price = 200 →
  ice_cream_price = 0.6 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.52 :=
by sorry

end NUMINAMATH_CALUDE_sundae_price_l2047_204794


namespace NUMINAMATH_CALUDE_cos_two_beta_l2047_204700

theorem cos_two_beta (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 7) (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  Real.cos (2 * β) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_beta_l2047_204700


namespace NUMINAMATH_CALUDE_solve_equation_l2047_204724

-- Define the original equation
def original_equation (x a : ℚ) : Prop :=
  (2 * x - 1) / 5 + 1 = (x + a) / 2

-- Define the incorrect equation after clearing denominators (with the mistake)
def incorrect_equation (x a : ℚ) : Prop :=
  2 * (2 * x - 1) + 1 = 5 * (x + a)

-- Theorem statement
theorem solve_equation :
  ∀ a : ℚ, 
    (∃ x : ℚ, incorrect_equation x a ∧ x = 4) →
    (a = -1 ∧ ∃ x : ℚ, original_equation x (-1) ∧ x = 13) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l2047_204724


namespace NUMINAMATH_CALUDE_intersection_and_slope_l2047_204782

theorem intersection_and_slope (k : ℝ) :
  (∃ y : ℝ, -3 * 3 + y = k ∧ 3 + y = 8) →
  k = -4 ∧ 
  (∀ x y : ℝ, x + y = 8 → y = -x + 8) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_slope_l2047_204782


namespace NUMINAMATH_CALUDE_grasshopper_cannot_return_l2047_204769

def jump_sequence (n : ℕ) : ℕ := n

theorem grasshopper_cannot_return : 
  ∀ (x₀ y₀ x₂₂₂₂ y₂₂₂₂ : ℤ),
  (x₀ + y₀) % 2 = 0 →
  (∀ n : ℕ, n ≤ 2222 → ∃ xₙ yₙ : ℤ, 
    (xₙ - x₀)^2 + (yₙ - y₀)^2 = (jump_sequence n)^2) →
  (x₂₂₂₂ + y₂₂₂₂) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_cannot_return_l2047_204769


namespace NUMINAMATH_CALUDE_sqrt_2y_lt_3y_iff_y_gt_2_div_9_l2047_204725

theorem sqrt_2y_lt_3y_iff_y_gt_2_div_9 :
  ∀ y : ℝ, y > 0 → (Real.sqrt (2 * y) < 3 * y ↔ y > 2 / 9) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2y_lt_3y_iff_y_gt_2_div_9_l2047_204725


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l2047_204727

def pizza_shares (total : ℚ) (ali : ℚ) (bea : ℚ) (chris : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  let dan := total - (ali + bea + chris)
  (dan, ali, chris, bea)

theorem pizza_consumption_order (total : ℚ) :
  let (dan, ali, chris, bea) := pizza_shares total (1/6) (1/8) (1/7)
  dan > ali ∧ ali > chris ∧ chris > bea := by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l2047_204727


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2047_204742

/-- The imaginary part of (1-i)^2 / (1+i) is -1 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I)^2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2047_204742


namespace NUMINAMATH_CALUDE_average_salary_all_employees_l2047_204759

theorem average_salary_all_employees
  (officer_avg_salary : ℕ)
  (non_officer_avg_salary : ℕ)
  (num_officers : ℕ)
  (num_non_officers : ℕ)
  (h1 : officer_avg_salary = 450)
  (h2 : non_officer_avg_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 495) :
  (officer_avg_salary * num_officers + non_officer_avg_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by sorry

end NUMINAMATH_CALUDE_average_salary_all_employees_l2047_204759


namespace NUMINAMATH_CALUDE_count_divisors_252_not_div_by_seven_l2047_204731

def divisors_not_div_by_seven (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0 ∧ x % 7 ≠ 0)

theorem count_divisors_252_not_div_by_seven :
  (divisors_not_div_by_seven 252).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_252_not_div_by_seven_l2047_204731


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2047_204765

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2047_204765


namespace NUMINAMATH_CALUDE_divisibility_by_43_l2047_204763

theorem divisibility_by_43 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  43 ∣ (7^p - 6^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_43_l2047_204763


namespace NUMINAMATH_CALUDE_equation_solutions_l2047_204757

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2047_204757


namespace NUMINAMATH_CALUDE_odd_periodic_two_at_one_l2047_204729

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x + 2) = f(x) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

/-- For a function f: ℝ → ℝ, if f is odd and has a period of 2, then f(1) = 0 -/
theorem odd_periodic_two_at_one (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_two_at_one_l2047_204729


namespace NUMINAMATH_CALUDE_correct_product_l2047_204776

theorem correct_product (a b : ℕ+) 
  (h1 : (a - 6) * b = 255) 
  (h2 : (a + 10) * b = 335) : 
  a * b = 285 := by sorry

end NUMINAMATH_CALUDE_correct_product_l2047_204776


namespace NUMINAMATH_CALUDE_simplify_expression_l2047_204718

theorem simplify_expression : (5 + 7 + 3 - 2) / 3 - 1 / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2047_204718


namespace NUMINAMATH_CALUDE_smallest_k_for_digit_sum_l2047_204752

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The product of 9 and (10^k - 1) -/
def special_product (k : ℕ) : ℕ := 9 * (10^k - 1)

/-- The statement to prove -/
theorem smallest_k_for_digit_sum : 
  (∀ k < 167, sum_of_digits (special_product k) < 1500) ∧ 
  sum_of_digits (special_product 167) ≥ 1500 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_digit_sum_l2047_204752


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l2047_204762

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    prove that the third term is 3/4. -/
theorem third_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (h1 : 0 < (1 : ℝ) - (1/4 : ℝ)) -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - (1/4 : ℝ)) = 16) -- Sum formula for infinite geometric series
  : a * (1/4)^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l2047_204762


namespace NUMINAMATH_CALUDE_modified_coin_expected_winnings_l2047_204703

/-- A coin with three possible outcomes -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  winnings_heads : ℚ
  winnings_tails : ℚ
  loss_edge : ℚ

/-- The modified weighted coin as described in the problem -/
def modified_coin : Coin :=
  { prob_heads := 1/3
  , prob_tails := 1/2
  , prob_edge := 1/6
  , winnings_heads := 2
  , winnings_tails := 2
  , loss_edge := 4 }

/-- Expected winnings from flipping the coin -/
def expected_winnings (c : Coin) : ℚ :=
  c.prob_heads * c.winnings_heads + c.prob_tails * c.winnings_tails - c.prob_edge * c.loss_edge

/-- Theorem stating that the expected winnings from flipping the modified coin is 1 -/
theorem modified_coin_expected_winnings :
  expected_winnings modified_coin = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_coin_expected_winnings_l2047_204703


namespace NUMINAMATH_CALUDE_line_up_five_people_youngest_not_ends_l2047_204796

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_youngest_at_ends (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem line_up_five_people_youngest_not_ends : 
  number_of_arrangements 5 - arrangements_with_youngest_at_ends 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_up_five_people_youngest_not_ends_l2047_204796


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2047_204785

theorem complex_arithmetic_equality : (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + 2 * Complex.I * (3 - 5 * Complex.I) = 11 - 9 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2047_204785


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2047_204784

theorem fraction_evaluation : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2047_204784


namespace NUMINAMATH_CALUDE_tower_arrangements_l2047_204712

/-- The number of ways to arrange n distinct objects --/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ways to arrange k objects from n distinct objects --/
def permutation (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k objects from n objects with repetition allowed --/
def multisetPermutation (n : ℕ) (k : List ℕ) : ℕ := sorry

/-- The number of different towers that can be built --/
def numTowers : ℕ := sorry

theorem tower_arrangements :
  let red := 3
  let blue := 3
  let green := 4
  let towerHeight := 9
  numTowers = multisetPermutation towerHeight [red - 1, blue, green] +
              multisetPermutation towerHeight [red, blue - 1, green] +
              multisetPermutation towerHeight [red, blue, green - 1] :=
by sorry

end NUMINAMATH_CALUDE_tower_arrangements_l2047_204712
