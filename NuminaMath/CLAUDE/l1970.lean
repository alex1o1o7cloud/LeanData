import Mathlib

namespace NUMINAMATH_CALUDE_quiz_average_change_l1970_197004

theorem quiz_average_change (total_students : ℕ) (dropped_score : ℝ) (new_average : ℝ) :
  total_students = 16 →
  dropped_score = 55 →
  new_average = 63 →
  (((total_students : ℝ) * new_average + dropped_score) / (total_students : ℝ)) = 62.5 :=
by sorry

end NUMINAMATH_CALUDE_quiz_average_change_l1970_197004


namespace NUMINAMATH_CALUDE_max_value_xyz_l1970_197099

/-- Given real numbers x, y, and z that are non-negative and satisfy the equation
    2x + 3xy² + 2z = 36, the maximum value of x²y²z is 144. -/
theorem max_value_xyz (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
    (h_eq : 2*x + 3*x*y^2 + 2*z = 36) :
    x^2 * y^2 * z ≤ 144 :=
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1970_197099


namespace NUMINAMATH_CALUDE_ladybugs_per_leaf_l1970_197073

theorem ladybugs_per_leaf (total_leaves : ℕ) (total_ladybugs : ℕ) (h1 : total_leaves = 84) (h2 : total_ladybugs = 11676) :
  total_ladybugs / total_leaves = 139 :=
by sorry

end NUMINAMATH_CALUDE_ladybugs_per_leaf_l1970_197073


namespace NUMINAMATH_CALUDE_walking_cycling_speeds_l1970_197040

/-- Proves that given conditions result in specific walking and cycling speeds -/
theorem walking_cycling_speeds (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 2 →
  speed_ratio = 4 →
  time_difference = 1/3 →
  ∃ (walking_speed cycling_speed : ℝ),
    walking_speed = 4.5 ∧
    cycling_speed = 18 ∧
    cycling_speed = speed_ratio * walking_speed ∧
    distance / walking_speed - distance / cycling_speed = time_difference :=
by sorry

end NUMINAMATH_CALUDE_walking_cycling_speeds_l1970_197040


namespace NUMINAMATH_CALUDE_triangle_area_formulas_l1970_197097

theorem triangle_area_formulas (R r : ℝ) (A B C : ℝ) :
  let T := R * r * (Real.sin A + Real.sin B + Real.sin C)
  T = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_formulas_l1970_197097


namespace NUMINAMATH_CALUDE_jakes_birdhouse_depth_l1970_197081

/-- Calculates the depth of Jake's birdhouse given the dimensions of both birdhouses and their volume difference --/
theorem jakes_birdhouse_depth
  (sara_width : ℝ) (sara_height : ℝ) (sara_depth : ℝ)
  (jake_width : ℝ) (jake_height : ℝ)
  (volume_difference : ℝ)
  (h1 : sara_width = 1) -- 1 foot
  (h2 : sara_height = 2) -- 2 feet
  (h3 : sara_depth = 2) -- 2 feet
  (h4 : jake_width = 16) -- 16 inches
  (h5 : jake_height = 20) -- 20 inches
  (h6 : volume_difference = 1152) -- 1,152 cubic inches
  : ∃ (jake_depth : ℝ),
    jake_depth = 25.2 ∧
    (jake_width * jake_height * jake_depth) - (sara_width * sara_height * sara_depth * 12^3) = volume_difference :=
by sorry

end NUMINAMATH_CALUDE_jakes_birdhouse_depth_l1970_197081


namespace NUMINAMATH_CALUDE_unique_a_value_l1970_197053

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 
  (A a ∩ B).Nonempty ∧ 
  Set.Nonempty (A a ∩ B) ∧
  (A a ∩ C) = ∅ ∧
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1970_197053


namespace NUMINAMATH_CALUDE_selina_sold_two_shirts_l1970_197028

/-- Represents the store credit and pricing system --/
structure StoreCredit where
  pants_credit : ℕ
  shorts_credit : ℕ
  shirt_credit : ℕ
  jacket_credit : ℕ

/-- Represents the items Selina sold --/
structure ItemsSold where
  pants : ℕ
  shorts : ℕ
  jackets : ℕ

/-- Represents the items Selina purchased --/
structure ItemsPurchased where
  shirt1_price : ℕ
  shirt2_price : ℕ
  pants_price : ℕ

/-- Calculates the total store credit for non-shirt items --/
def nonShirtCredit (sc : StoreCredit) (is : ItemsSold) : ℕ :=
  sc.pants_credit * is.pants + sc.shorts_credit * is.shorts + sc.jacket_credit * is.jackets

/-- Calculates the total price of purchased items --/
def totalPurchasePrice (ip : ItemsPurchased) : ℕ :=
  ip.shirt1_price + ip.shirt2_price + ip.pants_price

/-- Applies discount and tax to the purchase price --/
def finalPurchasePrice (price : ℕ) (discount : ℚ) (tax : ℚ) : ℚ :=
  (price : ℚ) * (1 - discount) * (1 + tax)

/-- Main theorem: Proves that Selina sold 2 shirts --/
theorem selina_sold_two_shirts 
  (sc : StoreCredit)
  (is : ItemsSold)
  (ip : ItemsPurchased)
  (discount : ℚ)
  (tax : ℚ)
  (remaining_credit : ℕ)
  (h1 : sc = ⟨5, 3, 4, 7⟩)
  (h2 : is = ⟨3, 5, 2⟩)
  (h3 : ip = ⟨10, 12, 15⟩)
  (h4 : discount = 1/10)
  (h5 : tax = 1/20)
  (h6 : remaining_credit = 25) :
  ∃ (shirts_sold : ℕ), shirts_sold = 2 ∧
    (nonShirtCredit sc is + sc.shirt_credit * shirts_sold : ℚ) =
    finalPurchasePrice (totalPurchasePrice ip) discount tax + remaining_credit :=
sorry

end NUMINAMATH_CALUDE_selina_sold_two_shirts_l1970_197028


namespace NUMINAMATH_CALUDE_midpoint_vector_equation_l1970_197080

/-- Given two points P₁ and P₂ in ℝ², prove that the point P satisfying 
    the vector equation P₁P - PP₂ = 0 has coordinates (1, 1) -/
theorem midpoint_vector_equation (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (-1, 2) → P₂ = (3, 0) → (P.1 - P₁.1, P.2 - P₁.2) = (P₂.1 - P.1, P₂.2 - P.2) → 
  P = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_midpoint_vector_equation_l1970_197080


namespace NUMINAMATH_CALUDE_expression_simplification_l1970_197058

theorem expression_simplification (x : ℤ) (h1 : -2 < x) (h2 : x ≤ 2) (h3 : x = 2) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1970_197058


namespace NUMINAMATH_CALUDE_election_percentages_correct_l1970_197062

def votes : List Nat := [1136, 7636, 10628, 8562, 6490]

def total_votes : Nat := votes.sum

def percentage (votes : Nat) (total : Nat) : Float :=
  (votes.toFloat / total.toFloat) * 100

def percentages : List Float :=
  votes.map (λ v => percentage v total_votes)

theorem election_percentages_correct :
  percentages ≈ [3.20, 21.54, 29.98, 24.15, 18.30] := by
  sorry

end NUMINAMATH_CALUDE_election_percentages_correct_l1970_197062


namespace NUMINAMATH_CALUDE_william_tax_is_800_l1970_197077

/-- Represents the farm tax system in a village -/
structure FarmTaxSystem where
  total_tax : ℝ
  taxable_land_percentage : ℝ
  william_land_percentage : ℝ

/-- Calculates the farm tax paid by Mr. William -/
def william_tax (system : FarmTaxSystem) : ℝ :=
  system.total_tax * system.william_land_percentage

/-- Theorem stating that Mr. William's farm tax is $800 -/
theorem william_tax_is_800 (system : FarmTaxSystem) 
  (h1 : system.total_tax = 5000)
  (h2 : system.taxable_land_percentage = 0.6)
  (h3 : system.william_land_percentage = 0.16) : 
  william_tax system = 800 := by
  sorry


end NUMINAMATH_CALUDE_william_tax_is_800_l1970_197077


namespace NUMINAMATH_CALUDE_total_clothing_cost_l1970_197051

def shorts_cost : ℚ := 14.28
def jacket_cost : ℚ := 4.74

theorem total_clothing_cost : shorts_cost + jacket_cost = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_cost_l1970_197051


namespace NUMINAMATH_CALUDE_older_friend_age_l1970_197015

theorem older_friend_age (younger_age older_age : ℕ) : 
  older_age - younger_age = 2 →
  younger_age + older_age = 74 →
  older_age = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_older_friend_age_l1970_197015


namespace NUMINAMATH_CALUDE_scaled_triangle_not_valid_l1970_197006

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The original triangle PQR -/
def original_triangle : Triangle :=
  { a := 15, b := 20, c := 25 }

/-- The scaled triangle PQR -/
def scaled_triangle : Triangle :=
  { a := 3 * original_triangle.a,
    b := 2 * original_triangle.b,
    c := original_triangle.c }

/-- Theorem stating that the scaled triangle is not valid -/
theorem scaled_triangle_not_valid :
  ¬(is_valid_triangle scaled_triangle) :=
sorry

end NUMINAMATH_CALUDE_scaled_triangle_not_valid_l1970_197006


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1970_197034

theorem price_reduction_percentage (original_price reduction_amount : ℝ) :
  original_price = 500 →
  reduction_amount = 400 →
  (reduction_amount / original_price) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1970_197034


namespace NUMINAMATH_CALUDE_everett_work_weeks_l1970_197092

/-- Given that Everett worked 5 hours every day and a total of 140 hours,
    prove that he worked for 4 weeks. -/
theorem everett_work_weeks :
  let hours_per_day : ℕ := 5
  let total_hours : ℕ := 140
  let days_per_week : ℕ := 7
  let hours_per_week : ℕ := hours_per_day * days_per_week
  total_hours / hours_per_week = 4 := by
sorry

end NUMINAMATH_CALUDE_everett_work_weeks_l1970_197092


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_sequences_l1970_197095

/-- Given a triangle ABC where:
    - The angles A, B, C form an arithmetic sequence
    - The sides a, b, c (opposite to angles A, B, C respectively) form a geometric sequence
    Prove that the triangle is equilateral -/
theorem equilateral_triangle_from_sequences (A B C a b c : ℝ) : 
  (∃ d : ℝ, B - A = d ∧ C - B = d) →  -- Angles form arithmetic sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Sides form geometric sequence
  A + B + C = π →                     -- Sum of angles in a triangle
  A > 0 ∧ B > 0 ∧ C > 0 →             -- Positive angles
  a > 0 ∧ b > 0 ∧ c > 0 →             -- Positive side lengths
  (A = π/3 ∧ B = π/3 ∧ C = π/3) :=    -- Triangle is equilateral
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_sequences_l1970_197095


namespace NUMINAMATH_CALUDE_newspaper_photos_theorem_l1970_197087

/-- Represents the number of photos in a section of the newspaper --/
def photos_in_section (pages : ℕ) (photos_per_page : ℕ) : ℕ :=
  pages * photos_per_page

/-- Calculates the total number of photos in the newspaper for a given day --/
def total_photos_per_day (section_a : ℕ) (section_b : ℕ) (section_c : ℕ) : ℕ :=
  section_a + section_b + section_c

theorem newspaper_photos_theorem :
  let section_a := photos_in_section 25 4
  let section_b := photos_in_section 18 6
  let section_c_monday := photos_in_section 12 5
  let section_c_tuesday := photos_in_section 15 3
  let monday_total := total_photos_per_day section_a section_b section_c_monday
  let tuesday_total := total_photos_per_day section_a section_b section_c_tuesday
  monday_total + tuesday_total = 521 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_theorem_l1970_197087


namespace NUMINAMATH_CALUDE_average_score_proof_l1970_197074

def student_A_score : ℚ := 92
def student_B_score : ℚ := 75
def student_C_score : ℚ := 98

def number_of_students : ℚ := 3

def average_score : ℚ := (student_A_score + student_B_score + student_C_score) / number_of_students

theorem average_score_proof : average_score = 88.3333333333333 := by
  sorry

end NUMINAMATH_CALUDE_average_score_proof_l1970_197074


namespace NUMINAMATH_CALUDE_largest_sum_simplification_l1970_197086

theorem largest_sum_simplification :
  let sums := [1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/5, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/5) ∧ (1/3 + 1/5 = 8/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_simplification_l1970_197086


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1970_197019

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l1970_197019


namespace NUMINAMATH_CALUDE_inequality_proof_l1970_197060

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + 4*b^2)) + (b^3 / (b^2 + 4*c^2)) + (c^3 / (c^2 + 4*a^2)) ≥ (a + b + c) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1970_197060


namespace NUMINAMATH_CALUDE_badge_exchange_l1970_197000

theorem badge_exchange (V T : ℕ) : 
  V = T + 5 ∧ 
  (V - V * 24 / 100 + T * 20 / 100 : ℚ) = (T - T * 20 / 100 + V * 24 / 100 : ℚ) - 1 →
  V = 50 ∧ T = 45 :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l1970_197000


namespace NUMINAMATH_CALUDE_range_of_m_l1970_197012

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (1/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem range_of_m :
  (∀ x, p x → q m x) ↔ (m ≥ 3/4 ∨ m ≤ -3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1970_197012


namespace NUMINAMATH_CALUDE_geometry_problem_l1970_197071

-- Define the lines and points
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 6*x - 8*y + 3 = 0
def result_line (x y : ℝ) : Prop := 4*x + 3*y - 18 = 0
def point_C : ℝ × ℝ := (-1, 1)
def point_D : ℝ × ℝ := (1, 3)

-- Define the circle
def result_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 10

theorem geometry_problem :
  -- Part 1: Line properties
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ result_line x y) ∧
  (∀ x y : ℝ, (6*x - 8*y) * (4*x + 3*y) = -48) ∧
  -- Part 2: Circle properties
  result_circle point_C.1 point_C.2 ∧
  result_circle point_D.1 point_D.2 ∧
  (∃ a : ℝ, result_circle a 0) :=
by sorry

end NUMINAMATH_CALUDE_geometry_problem_l1970_197071


namespace NUMINAMATH_CALUDE_tattoo_ratio_l1970_197096

def jason_arm_tattoos : ℕ := 2
def jason_leg_tattoos : ℕ := 3
def jason_arms : ℕ := 2
def jason_legs : ℕ := 2
def adam_tattoos : ℕ := 23

def jason_total_tattoos : ℕ :=
  jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

theorem tattoo_ratio :
  ∃ (m : ℕ), adam_tattoos = m * jason_total_tattoos + 3 ∧
  adam_tattoos.gcd jason_total_tattoos = 1 := by
  sorry

end NUMINAMATH_CALUDE_tattoo_ratio_l1970_197096


namespace NUMINAMATH_CALUDE_sin_six_arcsin_one_third_l1970_197027

theorem sin_six_arcsin_one_third :
  Real.sin (6 * Real.arcsin (1/3)) = 191 * Real.sqrt 2 / 729 := by
  sorry

end NUMINAMATH_CALUDE_sin_six_arcsin_one_third_l1970_197027


namespace NUMINAMATH_CALUDE_problem_statement_l1970_197069

theorem problem_statement : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1970_197069


namespace NUMINAMATH_CALUDE_simplify_expression_l1970_197009

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = -24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1970_197009


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1970_197033

theorem complex_fraction_equality : (5 * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1970_197033


namespace NUMINAMATH_CALUDE_total_books_combined_l1970_197070

theorem total_books_combined (keith_books jason_books amanda_books sophie_books : ℕ)
  (h1 : keith_books = 20)
  (h2 : jason_books = 21)
  (h3 : amanda_books = 15)
  (h4 : sophie_books = 30) :
  keith_books + jason_books + amanda_books + sophie_books = 86 := by
sorry

end NUMINAMATH_CALUDE_total_books_combined_l1970_197070


namespace NUMINAMATH_CALUDE_calculation_proof_l1970_197002

theorem calculation_proof : 2456 + 144 / 12 * 5 - 256 = 2260 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1970_197002


namespace NUMINAMATH_CALUDE_smallest_number_is_2544_l1970_197043

def is_smallest_number (x : ℕ) : Prop :=
  (x - 24) % 5 = 0 ∧
  (x - 24) % 10 = 0 ∧
  (x - 24) % 15 = 0 ∧
  (x - 24) / (Nat.lcm 5 (Nat.lcm 10 15)) = 84 ∧
  ∀ y, y < x → ¬(is_smallest_number y)

theorem smallest_number_is_2544 :
  is_smallest_number 2544 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_is_2544_l1970_197043


namespace NUMINAMATH_CALUDE_weight_of_steel_ingot_l1970_197094

/-- Given a weight vest and purchase conditions, prove the weight of each steel ingot. -/
theorem weight_of_steel_ingot
  (original_weight : ℝ)
  (weight_increase_percent : ℝ)
  (ingot_cost : ℝ)
  (discount_percent : ℝ)
  (final_cost : ℝ)
  (h1 : original_weight = 60)
  (h2 : weight_increase_percent = 0.60)
  (h3 : ingot_cost = 5)
  (h4 : discount_percent = 0.20)
  (h5 : final_cost = 72)
  : ∃ (num_ingots : ℕ), 
    num_ingots > 10 ∧ 
    (2 : ℝ) = (original_weight * weight_increase_percent) / num_ingots :=
by sorry

end NUMINAMATH_CALUDE_weight_of_steel_ingot_l1970_197094


namespace NUMINAMATH_CALUDE_cube_root_nested_expression_l1970_197042

theorem cube_root_nested_expression : 
  (2 * (2 * 8^(1/3))^(1/3))^(1/3) = 2^(5/9) := by sorry

end NUMINAMATH_CALUDE_cube_root_nested_expression_l1970_197042


namespace NUMINAMATH_CALUDE_sum_of_integers_l1970_197066

theorem sum_of_integers : (47 : ℤ) + (-27 : ℤ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1970_197066


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1970_197044

/-- The distance between the foci of the ellipse x^2 + 9y^2 = 324 is 24√2 -/
theorem ellipse_foci_distance : 
  let ellipse_equation := fun (x y : ℝ) => x^2 + 9*y^2 = 324
  ∃ f₁ f₂ : ℝ × ℝ, 
    (∀ x y, ellipse_equation x y → ((x - f₁.1)^2 + (y - f₁.2)^2) + ((x - f₂.1)^2 + (y - f₂.2)^2) = 2 * 324) ∧ 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (24 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1970_197044


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l1970_197076

theorem square_side_lengths_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l1970_197076


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1970_197001

/-- The number of sheep on the Stewart farm -/
def num_sheep : ℕ := 24

/-- The number of horses on the Stewart farm -/
def num_horses : ℕ := 56

/-- The ratio of sheep to horses -/
def sheep_to_horse_ratio : ℚ := 3 / 7

/-- The amount of food each horse eats per day in ounces -/
def horse_food_per_day : ℕ := 230

/-- The total amount of horse food needed per day in ounces -/
def total_horse_food_per_day : ℕ := 12880

theorem stewart_farm_sheep_count :
  (num_sheep : ℚ) / num_horses = sheep_to_horse_ratio ∧
  num_horses * horse_food_per_day = total_horse_food_per_day ∧
  num_sheep = 24 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1970_197001


namespace NUMINAMATH_CALUDE_sequence_properties_l1970_197037

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (is_arithmetic_sequence a ∧ 
   (is_geometric_sequence (a 4) (a 7) (a 9) → 
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1970_197037


namespace NUMINAMATH_CALUDE_not_divisible_by_four_sum_of_digits_l1970_197093

def numbers : List Nat := [3674, 3684, 3694, 3704, 3714, 3722]

theorem not_divisible_by_four_sum_of_digits : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    (n % 10 + (n / 10) % 10 = 11) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_sum_of_digits_l1970_197093


namespace NUMINAMATH_CALUDE_distinct_integers_product_sum_l1970_197050

theorem distinct_integers_product_sum (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80 →
  p + q + r + s + t = 36 := by
sorry

end NUMINAMATH_CALUDE_distinct_integers_product_sum_l1970_197050


namespace NUMINAMATH_CALUDE_michaels_weight_loss_goal_l1970_197003

/-- The total weight Michael wants to lose by June -/
def total_weight_loss (march_loss april_loss may_loss : ℕ) : ℕ :=
  march_loss + april_loss + may_loss

/-- Proof that Michael's total weight loss goal is 10 pounds -/
theorem michaels_weight_loss_goal :
  ∃ (march_loss april_loss may_loss : ℕ),
    march_loss = 3 ∧
    april_loss = 4 ∧
    may_loss = 3 ∧
    total_weight_loss march_loss april_loss may_loss = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_weight_loss_goal_l1970_197003


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1970_197057

theorem smaller_root_of_equation : 
  ∃ (x : ℚ), (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ 
  x = 5/6 ∧ 
  ∀ y, ((y - 5/6)^2 + (y - 5/6)*(y - 2/3) = 0 → y ≥ 5/6) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1970_197057


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1970_197090

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1970_197090


namespace NUMINAMATH_CALUDE_sequence_sum_eq_square_l1970_197085

def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (List.range n).sum

theorem sequence_sum_eq_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_eq_square_l1970_197085


namespace NUMINAMATH_CALUDE_parabola_segment_length_l1970_197029

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Length of PQ on a parabola -/
theorem parabola_segment_length
  (p : ℝ)
  (hp : p > 0)
  (P Q : ParabolaPoint)
  (hP : P.y^2 = 2*p*P.x)
  (hQ : Q.y^2 = 2*p*Q.x)
  (h_sum : P.x + Q.x = 3*p) :
  |P.x - Q.x| + p = 4*p :=
by sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l1970_197029


namespace NUMINAMATH_CALUDE_range_of_trigonometric_function_l1970_197079

theorem range_of_trigonometric_function :
  ∀ x : ℝ, 0 ≤ Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ∧
           Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trigonometric_function_l1970_197079


namespace NUMINAMATH_CALUDE_rational_fraction_implication_l1970_197041

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, (x / (x^2 + x + 1) : ℝ) = a) →
  (∃ b : ℚ, (x^2 / (x^4 + x^2 + 1) : ℝ) = b) :=
by sorry

end NUMINAMATH_CALUDE_rational_fraction_implication_l1970_197041


namespace NUMINAMATH_CALUDE_range_of_y_l1970_197024

theorem range_of_y (x y : ℝ) (h1 : x = 4 - y) (h2 : -2 ≤ x ∧ x ≤ -1) : 5 ≤ y ∧ y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_l1970_197024


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l1970_197030

/-- The number of minutes in a day -/
def minutes_per_day : ℚ := 24 * 60

/-- The number of minutes the watch loses per day -/
def minutes_lost_per_day : ℚ := 5/2

/-- The number of days between March 15 at 1 PM and March 21 at 9 AM -/
def days_elapsed : ℚ := 5 + 5/6

/-- The correct additional minutes to set the watch -/
def n : ℚ := 14 + 14/23

theorem watch_correction_theorem :
  n = (minutes_per_day / (minutes_per_day - minutes_lost_per_day) - 1) * (days_elapsed * minutes_per_day) :=
by sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l1970_197030


namespace NUMINAMATH_CALUDE_statue_cost_proof_l1970_197013

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 0.25 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 496 :=
by sorry

end NUMINAMATH_CALUDE_statue_cost_proof_l1970_197013


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1970_197061

theorem equation_solution_exists : ∃ (x y z : ℕ+), x + y + z + 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1970_197061


namespace NUMINAMATH_CALUDE_machine_subtraction_l1970_197005

theorem machine_subtraction (initial : ℕ) (added : ℕ) (subtracted : ℕ) (result : ℕ) :
  initial = 26 →
  added = 15 →
  result = 35 →
  initial + added - subtracted = result →
  subtracted = 6 := by
sorry

end NUMINAMATH_CALUDE_machine_subtraction_l1970_197005


namespace NUMINAMATH_CALUDE_distribution_of_four_men_five_women_l1970_197018

/-- The number of ways to distribute men and women into groups -/
def group_distribution (men women : ℕ) : ℕ :=
  let group_of_two := men.choose 1 * women.choose 1
  let group_of_three_1 := (men - 1).choose 2 * (women - 1).choose 1
  let group_of_three_2 := 1 * (women - 2).choose 2
  (group_of_two * group_of_three_1 * group_of_three_2) / 2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_of_four_men_five_women :
  group_distribution 4 5 = 360 := by
  sorry

#eval group_distribution 4 5

end NUMINAMATH_CALUDE_distribution_of_four_men_five_women_l1970_197018


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_two_l1970_197016

def A (m : ℝ) : Set ℝ := {-2, 3, 4*m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_two (m : ℝ) :
  B m ⊆ A m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_two_l1970_197016


namespace NUMINAMATH_CALUDE_quadratic_equiv_abs_value_l1970_197064

theorem quadratic_equiv_abs_value : ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ |x - 8| = 3) ↔ (b = -16 ∧ c = 55) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equiv_abs_value_l1970_197064


namespace NUMINAMATH_CALUDE_library_visitors_l1970_197072

theorem library_visitors (sunday_avg : ℕ) (month_days : ℕ) (month_avg : ℕ) :
  sunday_avg = 500 →
  month_days = 30 →
  month_avg = 200 →
  let sundays := (month_days + 6) / 7
  let other_days := month_days - sundays
  let other_avg := (month_days * month_avg - sundays * sunday_avg) / other_days
  other_avg = 140 := by
sorry

#eval (30 + 6) / 7  -- Should output 5, representing the number of Sundays

end NUMINAMATH_CALUDE_library_visitors_l1970_197072


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1970_197039

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, (2*a + 5)*x + (a - 2)*y + 4 = 0 ∧ (2 - a)*x + (a + 3)*y - 1 = 0 → 
    ((2*a + 5)*(2 - a) + (a - 2)*(a + 3) = 0)) → 
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1970_197039


namespace NUMINAMATH_CALUDE_prop_3_prop_4_l1970_197048

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersectionPP : Plane → Plane → Line)
variable (intersectionPL : Plane → Line → Prop)

-- Proposition ③
theorem prop_3 (α β γ : Plane) (m : Line) :
  perpendicularPP α β →
  perpendicularPP α γ →
  intersectionPP β γ = m →
  perpendicularPL α m :=
sorry

-- Proposition ④
theorem prop_4 (α β : Plane) (m n : Line) :
  perpendicularPL α m →
  perpendicularPL β n →
  perpendicular m n →
  perpendicularPP α β :=
sorry

end NUMINAMATH_CALUDE_prop_3_prop_4_l1970_197048


namespace NUMINAMATH_CALUDE_block_with_t_hole_difference_l1970_197023

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the dimensions and position of a T-shaped hole -/
structure THole where
  height : ℕ
  length : ℕ
  width : ℕ
  distanceFromFront : ℕ

/-- Calculates the number of cubes needed to create a block with a T-shaped hole -/
def cubesNeededWithHole (block : BlockDimensions) (hole : THole) : ℕ :=
  block.length * block.width * block.depth - (hole.height * hole.length + hole.width - 1)

/-- Theorem stating that a 7x7x6 block with the given T-shaped hole requires 3 fewer cubes -/
theorem block_with_t_hole_difference :
  let block := BlockDimensions.mk 7 7 6
  let hole := THole.mk 1 3 2 3
  block.length * block.width * block.depth - cubesNeededWithHole block hole = 3 := by
  sorry


end NUMINAMATH_CALUDE_block_with_t_hole_difference_l1970_197023


namespace NUMINAMATH_CALUDE_sales_after_three_years_l1970_197067

/-- The number of televisions sold initially -/
def initial_sales : ℕ := 327

/-- The annual increase rate as a percentage -/
def increase_rate : ℚ := 20 / 100

/-- The number of years for which the sales increase -/
def years : ℕ := 3

/-- Function to calculate sales after a given number of years -/
def sales_after_years (initial : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

/-- Theorem stating that the sales after 3 years is approximately 565 -/
theorem sales_after_three_years :
  ∃ ε > 0, |sales_after_years initial_sales increase_rate years - 565| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_after_three_years_l1970_197067


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l1970_197054

/-- Theorem: For a hyperbola with given conditions, a = 1 and b = 4 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a - y^2 / b = 1) →  -- Hyperbola equation
  (∃ k, ∀ x y, 2*x + y = 0 → y = k*x) →  -- One asymptote
  (∃ x y, x^2 + y^2 = 5 ∧ y = 0) →  -- One focus
  a = 1 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l1970_197054


namespace NUMINAMATH_CALUDE_expression_evaluation_l1970_197049

theorem expression_evaluation (x c : ℝ) (hx : x = 3) (hc : c = 2) :
  (x^2 + c)^2 - (x^2 - c)^2 = 72 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1970_197049


namespace NUMINAMATH_CALUDE_log_ratio_identity_l1970_197063

theorem log_ratio_identity (a b x : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hb : b > 0) (hx : x > 0) :
  (Real.log x / Real.log a) / (Real.log x / Real.log (a * b)) = 1 + Real.log b / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_identity_l1970_197063


namespace NUMINAMATH_CALUDE_evaluate_expression_l1970_197025

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1970_197025


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1970_197055

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ R / 100 * P * 10 = 4/5 * P ∧ R = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1970_197055


namespace NUMINAMATH_CALUDE_solve_equation_l1970_197026

theorem solve_equation : ∃ x : ℚ, (3 * x) / 7 = 6 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1970_197026


namespace NUMINAMATH_CALUDE_kids_total_savings_l1970_197011

-- Define the conversion rate
def pound_to_dollar : ℝ := 1.38

-- Define the savings for each child
def teagan_savings : ℝ := 200 * 0.01 + 15 * 1.00
def rex_savings : ℝ := 100 * 0.05 + 45 * 0.25 + 8 * pound_to_dollar
def toni_savings : ℝ := 330 * 0.10 + 12 * 5.00

-- Define the total savings
def total_savings : ℝ := teagan_savings + rex_savings + toni_savings

-- Theorem statement
theorem kids_total_savings : total_savings = 137.29 := by
  sorry

end NUMINAMATH_CALUDE_kids_total_savings_l1970_197011


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l1970_197020

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l1970_197020


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1970_197038

theorem algebraic_expression_value (a b c : ℝ) : 
  (∀ x, (x - 1) * (x + 2) = a * x^2 + b * x + c) → 
  4 * a - 2 * b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1970_197038


namespace NUMINAMATH_CALUDE_hillarys_descending_rate_l1970_197007

/-- Proof of Hillary's descending rate on Mt. Everest --/
theorem hillarys_descending_rate 
  (total_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_short : ℝ) 
  (total_time : ℝ) :
  total_distance = 4700 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_short = 700 →
  total_time = 6 →
  ∃ (hillary_descending_rate : ℝ),
    hillary_descending_rate = 1000 ∧
    hillary_descending_rate * (total_time - (total_distance - hillary_stop_short) / hillary_climbing_rate) = 
    (total_distance - hillary_stop_short) - (eddy_climbing_rate * total_time) :=
by sorry

end NUMINAMATH_CALUDE_hillarys_descending_rate_l1970_197007


namespace NUMINAMATH_CALUDE_f_composition_negative_three_eq_pi_l1970_197089

/-- Piecewise function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2
  else if x = 0 then Real.pi
  else 0

/-- Theorem stating that f(f(-3)) = π -/
theorem f_composition_negative_three_eq_pi : f (f (-3)) = Real.pi := by sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_eq_pi_l1970_197089


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_value_l1970_197046

/-- Given vectors a and b in R², if the magnitude of their sum equals the magnitude of their difference, then the second component of b is 3. -/
theorem vector_equality_implies_x_value (a b : ℝ × ℝ) :
  a = (2, -4) →
  b.1 = 6 →
  ‖a + b‖ = ‖a - b‖ →
  b.2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_equality_implies_x_value_l1970_197046


namespace NUMINAMATH_CALUDE_range_of_k_l1970_197098

-- Define the equation
def equation (x k : ℝ) : Prop := |x| / (x - 2) = k * x

-- Define the property of having three distinct real roots
def has_three_distinct_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ k ∧ equation x₂ k ∧ equation x₃ k

-- Theorem statement
theorem range_of_k (k : ℝ) :
  has_three_distinct_roots k ↔ 0 < k ∧ k < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1970_197098


namespace NUMINAMATH_CALUDE_prism_21_edges_9_faces_l1970_197036

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 21 edges has 9 faces. -/
theorem prism_21_edges_9_faces :
  ∀ p : Prism, p.edges = 21 → num_faces p = 9 := by
  sorry


end NUMINAMATH_CALUDE_prism_21_edges_9_faces_l1970_197036


namespace NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l1970_197088

noncomputable def largest_solution (x : ℝ) : Prop :=
  (Real.log 10 / Real.log (10 * x^3)) + (Real.log 10 / Real.log (100 * x^4)) = -1 ∧
  ∀ y, (Real.log 10 / Real.log (10 * y^3)) + (Real.log 10 / Real.log (100 * y^4)) = -1 → y ≤ x

theorem largest_solution_reciprocal_sixth_power (x : ℝ) :
  largest_solution x → 1 / x^6 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l1970_197088


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l1970_197031

theorem greatest_integer_problem : 
  ∃ (m : ℕ), 
    (m < 150) ∧ 
    (∃ (a : ℕ), m = 9 * a - 2) ∧ 
    (∃ (b : ℕ), m = 5 * b + 4) ∧ 
    (∀ (n : ℕ), 
      (n < 150) → 
      (∃ (c : ℕ), n = 9 * c - 2) → 
      (∃ (d : ℕ), n = 5 * d + 4) → 
      n ≤ m) ∧
    m = 124 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l1970_197031


namespace NUMINAMATH_CALUDE_smallest_AC_l1970_197032

/-- Triangle ABC with point D on AC --/
structure Triangle :=
  (AC : ℕ)
  (CD : ℕ)

/-- Conditions for the triangle --/
def ValidTriangle (t : Triangle) : Prop :=
  t.AC > t.CD ∧ 
  2 * t.AC * t.CD = t.CD^2 + 57

/-- The theorem to prove --/
theorem smallest_AC : 
  ∀ t : Triangle, ValidTriangle t → t.AC ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_AC_l1970_197032


namespace NUMINAMATH_CALUDE_common_roots_solution_l1970_197014

/-- Two cubic polynomials with common roots -/
def poly1 (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 14*x + 8

def poly2 (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 17*x + 10

/-- The polynomials have two distinct common roots -/
def has_two_common_roots (a b : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧ poly1 a r = 0 ∧ poly1 a s = 0 ∧ poly2 b r = 0 ∧ poly2 b s = 0

/-- The main theorem -/
theorem common_roots_solution :
  has_two_common_roots 7 8 :=
sorry

end NUMINAMATH_CALUDE_common_roots_solution_l1970_197014


namespace NUMINAMATH_CALUDE_max_beauty_value_bound_l1970_197045

/-- Represents a figure with circles and segments arranged into pentagons -/
structure Figure where
  circles : Nat
  segments : Nat
  pentagons : Nat

/-- Represents a method of filling numbers in the circles -/
def FillingMethod := Fin 15 → Fin 3

/-- Calculates the beauty value of a filling method -/
def beautyValue (f : Figure) (m : FillingMethod) : Nat :=
  sorry

/-- The maximum possible beauty value -/
def maxBeautyValue (f : Figure) : Nat :=
  sorry

theorem max_beauty_value_bound (f : Figure) 
  (h1 : f.circles = 15) 
  (h2 : f.segments = 20) 
  (h3 : f.pentagons = 6) : 
  maxBeautyValue f ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_beauty_value_bound_l1970_197045


namespace NUMINAMATH_CALUDE_player_playing_time_l1970_197075

/-- Calculates the playing time for each player in a sports tournament. -/
theorem player_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_playing_time_l1970_197075


namespace NUMINAMATH_CALUDE_bridesmaids_count_l1970_197021

/-- Represents the makeup requirements for bridesmaids --/
structure MakeupRequirements where
  lipGlossPerTube : ℕ
  mascaraPerTube : ℕ
  lipGlossTubs : ℕ
  tubesPerLipGlossTub : ℕ
  mascaraTubs : ℕ
  tubesPerMascaraTub : ℕ

/-- Represents the makeup styles chosen by bridesmaids --/
inductive MakeupStyle
  | Glam
  | Natural

/-- Calculates the total number of bridesmaids given the makeup requirements --/
def totalBridesmaids (req : MakeupRequirements) : ℕ :=
  let totalLipGloss := req.lipGlossTubs * req.tubesPerLipGlossTub * req.lipGlossPerTube
  let totalMascara := req.mascaraTubs * req.tubesPerMascaraTub * req.mascaraPerTube
  let glamBridesmaids := totalLipGloss / 3  -- Each glam bridesmaid needs 2 lip gloss + 1 natural
  min glamBridesmaids (totalMascara / 2)  -- Each bridesmaid needs at least 1 mascara

/-- Proves that given the specific makeup requirements, there are 24 bridesmaids --/
theorem bridesmaids_count (req : MakeupRequirements) 
    (h1 : req.lipGlossPerTube = 3)
    (h2 : req.mascaraPerTube = 5)
    (h3 : req.lipGlossTubs = 6)
    (h4 : req.tubesPerLipGlossTub = 2)
    (h5 : req.mascaraTubs = 4)
    (h6 : req.tubesPerMascaraTub = 3) :
    totalBridesmaids req = 24 := by
  sorry

#eval totalBridesmaids { 
  lipGlossPerTube := 3, 
  mascaraPerTube := 5, 
  lipGlossTubs := 6, 
  tubesPerLipGlossTub := 2, 
  mascaraTubs := 4, 
  tubesPerMascaraTub := 3 
}

end NUMINAMATH_CALUDE_bridesmaids_count_l1970_197021


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1970_197035

theorem polynomial_division_remainder : ∃ (q : Polynomial ℝ), 
  x^6 - x^5 - x^4 + x^3 + x^2 - x = (x^2 - 4) * (x + 1) * q + (21*x^2 - 13*x - 32) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1970_197035


namespace NUMINAMATH_CALUDE_x_range_l1970_197022

theorem x_range (x : ℝ) (h1 : 1 / x ≤ 3) (h2 : 1 / x ≥ -2) : x ≥ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1970_197022


namespace NUMINAMATH_CALUDE_candy_distribution_l1970_197047

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 648) 
  (h2 : num_bags = 8) 
  (h3 : candy_per_bag * num_bags = total_candy) :
  candy_per_bag = 81 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1970_197047


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_one_l1970_197017

theorem complex_equality_implies_ratio_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^4 = -1 → (a + b * Complex.I)^4 = (a - b * Complex.I)^4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_one_l1970_197017


namespace NUMINAMATH_CALUDE_point_p_coordinates_and_b_range_l1970_197091

/-- The system of equations defining point P -/
def system_of_equations (x y a b : ℝ) : Prop :=
  x + y = 2*a - b - 4 ∧ x - y = b - 4

/-- Point P is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- There are only three integers that satisfy the requirements for a -/
def three_integers_for_a (a : ℝ) : Prop :=
  a = 1 ∨ a = 2 ∨ a = 3

theorem point_p_coordinates_and_b_range :
  (∀ x y : ℝ, system_of_equations x y 1 1 → x = -3 ∧ y = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, system_of_equations x y a b ∧ second_quadrant x y) →
    three_integers_for_a a → 0 ≤ b ∧ b < 1) :=
sorry

end NUMINAMATH_CALUDE_point_p_coordinates_and_b_range_l1970_197091


namespace NUMINAMATH_CALUDE_set_intersection_example_l1970_197008

theorem set_intersection_example :
  let A : Set ℤ := {1, 0, 3}
  let B : Set ℤ := {-1, 1, 2, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l1970_197008


namespace NUMINAMATH_CALUDE_inequality_proof_l1970_197052

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 2 * Real.sqrt 2 - 2) 
  (hb : b = Real.exp 2 / 7) 
  (hc : c = Real.log 2) : 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1970_197052


namespace NUMINAMATH_CALUDE_vector_operation_result_l1970_197084

/-- Prove that the vector operation (3, -6) - 5(1, -9) + (-1, 4) results in (-3, 43) -/
theorem vector_operation_result : 
  (⟨3, -6⟩ : ℝ × ℝ) - 5 • ⟨1, -9⟩ + ⟨-1, 4⟩ = ⟨-3, 43⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l1970_197084


namespace NUMINAMATH_CALUDE_al_sandwich_options_l1970_197010

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 4

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether ham is available. -/
def has_ham : Prop := True

/-- Represents whether chicken is available. -/
def has_chicken : Prop := True

/-- Represents whether cheddar cheese is available. -/
def has_cheddar : Prop := True

/-- Represents whether white bread is available. -/
def has_white_bread : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_bread

/-- Represents the number of sandwiches with white bread and chicken combination. -/
def white_chicken_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem al_sandwich_options : 
  num_bread * num_meat * num_cheese - ham_cheddar_combos - white_chicken_combos = 111 := by
  sorry

end NUMINAMATH_CALUDE_al_sandwich_options_l1970_197010


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1970_197056

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≥ b) (h5 : c^2 = a^2 + b^2) : 
  a + b / 2 > c ∧ c > 8 / 9 * (a + b / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1970_197056


namespace NUMINAMATH_CALUDE_prob_both_red_is_one_ninth_l1970_197068

/-- The probability of drawing a red ball from both bags A and B -/
def prob_both_red (red_a white_a red_b white_b : ℕ) : ℚ :=
  (red_a : ℚ) / (red_a + white_a) * (red_b : ℚ) / (red_b + white_b)

/-- Theorem: The probability of drawing a red ball from both Bag A and Bag B is 1/9 -/
theorem prob_both_red_is_one_ninth :
  prob_both_red 4 2 1 5 = 1 / 9 := by
  sorry

#eval prob_both_red 4 2 1 5

end NUMINAMATH_CALUDE_prob_both_red_is_one_ninth_l1970_197068


namespace NUMINAMATH_CALUDE_download_time_360GB_50MBps_l1970_197078

/-- Calculates the download time in hours for a given program size and download speed -/
def downloadTime (programSizeGB : ℕ) (downloadSpeedMBps : ℕ) : ℚ :=
  let programSizeMB := programSizeGB * 1000
  let downloadTimeSeconds := programSizeMB / downloadSpeedMBps
  downloadTimeSeconds / 3600

/-- Proves that downloading a 360 GB program at 50 MB/s takes 2 hours -/
theorem download_time_360GB_50MBps :
  downloadTime 360 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_download_time_360GB_50MBps_l1970_197078


namespace NUMINAMATH_CALUDE_equal_distribution_of_cards_l1970_197083

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_cards_l1970_197083


namespace NUMINAMATH_CALUDE_parabola_translation_l1970_197059

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := 2 * (x - 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = translated_parabola x ↔ y - 3 = original_parabola (x - 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l1970_197059


namespace NUMINAMATH_CALUDE_sampling_is_systematic_l1970_197065

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with rows and seats --/
structure Auditorium where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  auditorium : Auditorium
  seatNumberSelected : Nat

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.seatNumberSelected > 0 ∧ 
  strategy.seatNumberSelected ≤ strategy.auditorium.seatsPerRow ∧
  strategy.seatNumberSelected = strategy.seatNumberSelected

/-- Theorem stating that the given sampling strategy is systematic --/
theorem sampling_is_systematic (a : Auditorium) (s : SamplingStrategy) :
  a.rows = 25 → 
  a.seatsPerRow = 20 → 
  s.auditorium = a → 
  s.seatNumberSelected = 15 → 
  isSystematicSampling s := by
  sorry

#check sampling_is_systematic

end NUMINAMATH_CALUDE_sampling_is_systematic_l1970_197065


namespace NUMINAMATH_CALUDE_concert_attendance_l1970_197082

/-- Proves that given the initial ratio of women to men is 1:2, and after 12 women and 29 men left
    the ratio became 1:3, the original number of people at the concert was 21. -/
theorem concert_attendance (w m : ℕ) : 
  w / m = 1 / 2 →  -- Initial ratio of women to men
  (w - 12) / (m - 29) = 1 / 3 →  -- Ratio after some people left
  w + m = 21  -- Total number of people initially
  := by sorry

end NUMINAMATH_CALUDE_concert_attendance_l1970_197082
