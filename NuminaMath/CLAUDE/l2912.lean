import Mathlib

namespace NUMINAMATH_CALUDE_f_matches_table_l2912_291236

/-- The function that generates the output values -/
def f (n : ℕ) : ℕ := 2 * n - 1

/-- The proposition that the function f matches the given table for n from 1 to 5 -/
theorem f_matches_table : 
  f 1 = 1 ∧ f 2 = 3 ∧ f 3 = 5 ∧ f 4 = 7 ∧ f 5 = 9 := by
  sorry

#check f_matches_table

end NUMINAMATH_CALUDE_f_matches_table_l2912_291236


namespace NUMINAMATH_CALUDE_equivalent_division_l2912_291259

theorem equivalent_division (x : ℝ) :
  x / (4^3 / 8) * Real.sqrt (7 / 5) = x / ((8 * Real.sqrt 35) / 5) := by sorry

end NUMINAMATH_CALUDE_equivalent_division_l2912_291259


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2912_291288

theorem isosceles_triangle_vertex_angle (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a^2 = 3 * b * h →
  b = 2 * a * Real.cos (π / 4) →
  h = a * Real.sin (π / 4) →
  let vertex_angle := π - 2 * (π / 4)
  vertex_angle = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2912_291288


namespace NUMINAMATH_CALUDE_function_properties_l2912_291224

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- State the theorem
theorem function_properties (a b : ℝ) :
  a ≠ 0 →
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  (f a b 1 = 2 → a > 0 → b > 0 → 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 4/b' ≥ 9) ∧
    (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 9)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2912_291224


namespace NUMINAMATH_CALUDE_angle_equality_l2912_291283

theorem angle_equality (angle1 angle2 angle3 : ℝ) : 
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle2 + angle3 = 90) →  -- angle2 and angle3 are complementary
  (angle1 = 40) →           -- angle1 is 40 degrees
  (angle3 = 40) :=          -- conclusion: angle3 is 40 degrees
by
  sorry

#check angle_equality

end NUMINAMATH_CALUDE_angle_equality_l2912_291283


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l2912_291269

theorem inscribed_cylinder_height (r_hemisphere r_cylinder : ℝ) (h_hemisphere : r_hemisphere = 7) (h_cylinder : r_cylinder = 3) :
  let h_cylinder := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h_cylinder = Real.sqrt 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l2912_291269


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l2912_291229

open Real

theorem triangle_side_sum_range (A B C a b c : Real) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  cos B / b + cos C / c = 2 * sqrt 3 * sin A / (3 * sin C) →
  cos B + sqrt 3 * sin B = 2 →
  3 / 2 < a + c ∧ a + c ≤ sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l2912_291229


namespace NUMINAMATH_CALUDE_sequence_existence_l2912_291225

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)) ∧
    (a (n + 1) = a 1) ∧
    (a (n + 2) = a 2)) ↔ 
  (3 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l2912_291225


namespace NUMINAMATH_CALUDE_cricket_team_size_l2912_291266

theorem cricket_team_size :
  ∀ (n : ℕ) (captain_age wicket_keeper_age team_avg_age remaining_avg_age : ℝ),
    n > 0 →
    captain_age = 26 →
    wicket_keeper_age = captain_age + 3 →
    team_avg_age = 23 →
    remaining_avg_age = team_avg_age - 1 →
    team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age →
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2912_291266


namespace NUMINAMATH_CALUDE_circle_radii_formula_l2912_291234

/-- Given a triangle ABC with circumradius R and heights h_a, h_b, h_c,
    the radii t_a, t_b, t_c of circles tangent internally to the inscribed circle
    at vertices A, B, C and externally to each other satisfy the given formulas. -/
theorem circle_radii_formula (a b c R h_a h_b h_c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) :
  ∃ (t_a t_b t_c : ℝ),
    (t_a > 0 ∧ t_b > 0 ∧ t_c > 0) ∧
    (t_a = (R * h_a) / (a + h_a)) ∧
    (t_b = (R * h_b) / (b + h_b)) ∧
    (t_c = (R * h_c) / (c + h_c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_formula_l2912_291234


namespace NUMINAMATH_CALUDE_circle_max_sum_l2912_291241

theorem circle_max_sum :
  ∀ x y : ℤ, x^2 + y^2 = 16 → x + y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_l2912_291241


namespace NUMINAMATH_CALUDE_inequality_proof_l2912_291295

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (b + c) = b / (c + a) - c / (a + b)) :
  b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2912_291295


namespace NUMINAMATH_CALUDE_p_18_equals_negative_one_l2912_291215

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

/-- Theorem: For a quadratic function with given properties, p(18) = -1 -/
theorem p_18_equals_negative_one
  (d e f : ℝ)
  (p : ℝ → ℝ)
  (h_quad : p = QuadraticFunction d e f)
  (h_sym : p 6 = p 12)
  (h_max : IsLocalMax p 10)
  (h_p0 : p 0 = -1) :
  p 18 = -1 := by
  sorry

end NUMINAMATH_CALUDE_p_18_equals_negative_one_l2912_291215


namespace NUMINAMATH_CALUDE_factor_expression_l2912_291264

theorem factor_expression (b : ℝ) : 56 * b^2 + 168 * b = 56 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2912_291264


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2912_291208

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2912_291208


namespace NUMINAMATH_CALUDE_inequality_conditions_l2912_291281

theorem inequality_conditions (x y z : ℝ) 
  (h1 : y - x < 1.5 * Real.sqrt (x^2))
  (h2 : z = 2 * (y + x)) :
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conditions_l2912_291281


namespace NUMINAMATH_CALUDE_g_value_at_2_l2912_291228

def g (x : ℝ) : ℝ := 3 * x^8 - 4 * x^4 + 2 * x^2 - 6

theorem g_value_at_2 (h : g (-2) = 10) : g 2 = 1402 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_2_l2912_291228


namespace NUMINAMATH_CALUDE_same_name_existence_l2912_291265

/-- Represents a child in the class -/
structure Child where
  forename : Nat
  surname : Nat

/-- The problem statement -/
theorem same_name_existence 
  (children : Finset Child) 
  (h_count : children.card = 33) 
  (h_range : ∀ c ∈ children, c.forename ≤ 10 ∧ c.surname ≤ 10) 
  (h_appear : ∀ n : Nat, n ≤ 10 → 
    (∃ c ∈ children, c.forename = n) ∧ 
    (∃ c ∈ children, c.surname = n)) :
  ∃ c1 c2 : Child, c1 ∈ children ∧ c2 ∈ children ∧ c1 ≠ c2 ∧ 
    c1.forename = c2.forename ∧ c1.surname = c2.surname :=
sorry

end NUMINAMATH_CALUDE_same_name_existence_l2912_291265


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2912_291216

theorem intersection_of_sets :
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}
  A ∩ B = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2912_291216


namespace NUMINAMATH_CALUDE_initial_queue_size_l2912_291218

theorem initial_queue_size (n : ℕ) : 
  (∀ A : ℕ, A = 41 * n) →  -- Current total age
  (A + 69 = 45 * (n + 1)) → -- New total age after 7th person joins
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_queue_size_l2912_291218


namespace NUMINAMATH_CALUDE_tile_relationship_l2912_291287

theorem tile_relationship (r : ℕ) (w : ℕ) : 
  (3 ≤ r ∧ r ≤ 7) → 
  (
    (r = 3 ∧ w = 6) ∨
    (r = 4 ∧ w = 8) ∨
    (r = 5 ∧ w = 10) ∨
    (r = 6 ∧ w = 12) ∨
    (r = 7 ∧ w = 14)
  ) →
  w = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_tile_relationship_l2912_291287


namespace NUMINAMATH_CALUDE_inverse_direct_variation_l2912_291275

theorem inverse_direct_variation (k c : ℝ) (x y z : ℝ) : 
  (5 * y = k / (x ^ 2)) →
  (3 * z = c * x) →
  (5 * 25 = k / (2 ^ 2)) →
  (x = 4) →
  (z = 6) →
  (y = 6.25) := by
  sorry

end NUMINAMATH_CALUDE_inverse_direct_variation_l2912_291275


namespace NUMINAMATH_CALUDE_distance_after_movements_l2912_291292

/-- The distance between two points given a path with specific movements -/
theorem distance_after_movements (south west north east : ℝ) :
  south = 50 ∧ west = 80 ∧ north = 30 ∧ east = 10 →
  Real.sqrt ((south - north)^2 + (west - east)^2) = 50 * Real.sqrt 106 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_after_movements_l2912_291292


namespace NUMINAMATH_CALUDE_negation_equivalence_l2912_291211

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 4*x + 5 ≤ 0) ↔ (∀ x : ℝ, x^2 + 4*x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2912_291211


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2912_291293

theorem isosceles_triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 15 = 0 →
  x > 0 →
  x < 4 →
  2 + 2 + x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2912_291293


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2912_291246

/-- The complex number i such that i² = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number z -/
noncomputable def z : ℂ := (1 - i) / (1 + i) + 4 - 2*i

/-- Theorem stating that the magnitude of z is 5 -/
theorem magnitude_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2912_291246


namespace NUMINAMATH_CALUDE_seashells_count_l2912_291252

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l2912_291252


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l2912_291206

theorem factorization_of_polynomial (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l2912_291206


namespace NUMINAMATH_CALUDE_tuesday_toys_bought_l2912_291226

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := 5

/-- The number of dog toys Daisy had left on Tuesday after losing some -/
def tuesday_remaining : ℕ := 3

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_new : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_if_found : ℕ := 13

/-- The number of dog toys Daisy's owner bought on Tuesday -/
def tuesday_new : ℕ := total_if_found - tuesday_remaining - wednesday_new

theorem tuesday_toys_bought :
  tuesday_new = 5 :=
by sorry

end NUMINAMATH_CALUDE_tuesday_toys_bought_l2912_291226


namespace NUMINAMATH_CALUDE_slide_problem_l2912_291210

theorem slide_problem (initial_boys : ℕ) (total_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end NUMINAMATH_CALUDE_slide_problem_l2912_291210


namespace NUMINAMATH_CALUDE_picture_frame_problem_l2912_291202

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  wood_width : ℝ

/-- Calculates the area of the frame material -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.wood_width) * (f.outer_width - 2 * f.wood_width)

/-- Calculates the sum of the lengths of the four interior edges -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * (f.outer_length - 2 * f.wood_width) + 2 * (f.outer_width - 2 * f.wood_width)

theorem picture_frame_problem :
  ∀ f : Frame,
    f.wood_width = 2 →
    f.outer_length = 7 →
    frame_area f = 34 →
    interior_perimeter f = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_frame_problem_l2912_291202


namespace NUMINAMATH_CALUDE_radish_count_l2912_291204

theorem radish_count (total : ℕ) (difference : ℕ) (radishes : ℕ) : 
  total = 100 →
  difference = 24 →
  radishes = total - difference / 2 →
  radishes = 62 := by
sorry

end NUMINAMATH_CALUDE_radish_count_l2912_291204


namespace NUMINAMATH_CALUDE_regular_1001_gon_labeling_existence_l2912_291219

theorem regular_1001_gon_labeling_existence :
  ∃ f : Fin 1001 → Fin 1001,
    Function.Bijective f ∧
    ∀ (r : Fin 1001) (b : Bool),
      ∃ i : Fin 1001,
        f ((i + r) % 1001) = if b then i else (1001 - i) % 1001 := by
  sorry

end NUMINAMATH_CALUDE_regular_1001_gon_labeling_existence_l2912_291219


namespace NUMINAMATH_CALUDE_subset_of_countable_is_finite_or_countable_l2912_291242

theorem subset_of_countable_is_finite_or_countable 
  (X : Set α) (hX : Countable X) (A : Set α) (hA : A ⊆ X) :
  (Finite A) ∨ (Countable A) :=
sorry

end NUMINAMATH_CALUDE_subset_of_countable_is_finite_or_countable_l2912_291242


namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2912_291238

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) →
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : ∃ a b : ℕ, bob_pennies a b := by
  sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2912_291238


namespace NUMINAMATH_CALUDE_abc_inequality_l2912_291232

theorem abc_inequality (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2912_291232


namespace NUMINAMATH_CALUDE_set_membership_properties_l2912_291262

def A : Set Int := {x | ∃ k, x = 3 * k - 1}
def B : Set Int := {x | ∃ k, x = 3 * k + 1}
def C : Set Int := {x | ∃ k, x = 3 * k}

theorem set_membership_properties (a b c : Int) (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  (2 * a ∈ B) ∧ (2 * b ∈ A) ∧ (a + b ∈ C) := by
  sorry

end NUMINAMATH_CALUDE_set_membership_properties_l2912_291262


namespace NUMINAMATH_CALUDE_octal_to_binary_l2912_291291

-- Define the octal number
def octal_177 : ℕ := 177

-- Define the binary number
def binary_1111111 : ℕ := 127

-- Theorem statement
theorem octal_to_binary :
  (octal_177 : ℕ) = binary_1111111 := by sorry

end NUMINAMATH_CALUDE_octal_to_binary_l2912_291291


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2912_291212

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (∃ k1 : ℕ, 1657 = n * k1 + 6) ∧ 
  (∃ k2 : ℕ, 2037 = n * k2 + 5) ∧ 
  (∀ m : ℕ, (∃ j1 : ℕ, 1657 = m * j1 + 6) ∧ (∃ j2 : ℕ, 2037 = m * j2 + 5) → m ≤ n) →
  n = 127 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2912_291212


namespace NUMINAMATH_CALUDE_henry_pill_cost_l2912_291290

/-- Calculates the total cost of pills for Henry over 21 days -/
def totalPillCost (daysTotal : ℕ) (pillsPerDay : ℕ) (pillType1Count : ℕ) (pillType2Count : ℕ)
  (pillType1Cost : ℚ) (pillType2Cost : ℚ) (pillType3ExtraCost : ℚ) 
  (discountRate : ℚ) (priceIncrease : ℚ) : ℚ :=
  let pillType3Count := pillsPerDay - (pillType1Count + pillType2Count)
  let pillType3Cost := pillType2Cost + pillType3ExtraCost
  let regularDayCost := pillType1Count * pillType1Cost + pillType2Count * pillType2Cost + 
                        pillType3Count * pillType3Cost
  let discountDays := daysTotal / 3
  let regularDays := daysTotal - discountDays
  let discountDayCost := (1 - discountRate) * (pillType1Count * pillType1Cost + pillType2Count * pillType2Cost) +
                         pillType3Count * (pillType3Cost + priceIncrease)
  regularDays * regularDayCost + discountDays * discountDayCost

/-- The total cost of Henry's pills over 21 days is $1485.10 -/
theorem henry_pill_cost : 
  totalPillCost 21 12 4 5 (3/2) 7 3 (1/5) (5/2) = 1485.1 := by
  sorry

end NUMINAMATH_CALUDE_henry_pill_cost_l2912_291290


namespace NUMINAMATH_CALUDE_ages_ratio_three_to_one_l2912_291237

/-- Represents a person's age --/
structure Age where
  years : ℕ

/-- Represents the ages of Claire and Pete --/
structure AgesPair where
  claire : Age
  pete : Age

/-- The conditions of the problem --/
def problem_conditions (ages : AgesPair) : Prop :=
  (ages.claire.years - 3 = 2 * (ages.pete.years - 3)) ∧
  (ages.pete.years - 7 = (ages.claire.years - 7) / 4)

/-- The theorem to prove --/
theorem ages_ratio_three_to_one (ages : AgesPair) :
  problem_conditions ages →
  ∃ (claire_age pete_age : ℕ),
    claire_age = ages.claire.years - 6 ∧
    pete_age = ages.pete.years - 6 ∧
    3 * pete_age = claire_age :=
by
  sorry


end NUMINAMATH_CALUDE_ages_ratio_three_to_one_l2912_291237


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l2912_291286

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 5 = 25) → initial_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l2912_291286


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2912_291231

theorem sqrt_difference_equality : 
  Real.sqrt (9/2) - Real.sqrt (8/5) = (15 * Real.sqrt 2 - 4 * Real.sqrt 10) / 10 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2912_291231


namespace NUMINAMATH_CALUDE_garden_fencing_l2912_291233

/-- Calculates the perimeter of a rectangular garden with given length and width ratio --/
theorem garden_fencing (length : ℝ) (h1 : length = 80) : 
  2 * (length + length / 2) = 240 := by
  sorry


end NUMINAMATH_CALUDE_garden_fencing_l2912_291233


namespace NUMINAMATH_CALUDE_no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l2912_291223

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point on y = -4/x
theorem no_harmonic_point_on_reciprocal : ¬∃ x : ℝ, is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + 6 * x + c

theorem unique_harmonic_point :
  ∃! (a c : ℝ), a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (quadratic_function a c x)) ∧
  is_harmonic_point (5/2) (quadratic_function a c (5/2)) := by sorry

-- Part 3: Range of m for the modified quadratic function
def modified_quadratic (x : ℝ) : ℝ := -x^2 + 6*x - 6

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, 1 ≤ x → x ≤ m → -1 ≤ modified_quadratic x ∧ modified_quadratic x ≤ 3) ↔
  (3 ≤ m ∧ m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_no_harmonic_point_on_reciprocal_unique_harmonic_point_range_of_m_l2912_291223


namespace NUMINAMATH_CALUDE_correct_junior_teachers_in_sample_l2912_291213

/-- Represents the number of teachers in each category -/
structure TeacherPopulation where
  total : Nat
  junior : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  populationSize : Nat
  sampleSize : Nat
  juniorInPopulation : Nat
  juniorInSample : Nat

/-- Calculates the number of junior teachers in a stratified sample -/
def calculateJuniorTeachersInSample (pop : TeacherPopulation) (sampleSize : Nat) : Nat :=
  (pop.junior * sampleSize) / pop.total

/-- Theorem stating that the calculated number of junior teachers in the sample is correct -/
theorem correct_junior_teachers_in_sample (pop : TeacherPopulation) (sample : StratifiedSample) 
    (h1 : pop.total = 200)
    (h2 : pop.junior = 80)
    (h3 : sample.populationSize = pop.total)
    (h4 : sample.sampleSize = 50)
    (h5 : sample.juniorInPopulation = pop.junior)
    (h6 : sample.juniorInSample = calculateJuniorTeachersInSample pop sample.sampleSize) :
  sample.juniorInSample = 20 := by
  sorry

#check correct_junior_teachers_in_sample

end NUMINAMATH_CALUDE_correct_junior_teachers_in_sample_l2912_291213


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2912_291263

theorem imaginary_part_of_z (z : ℂ) (h : z + 3 - 4*I = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2912_291263


namespace NUMINAMATH_CALUDE_car_distribution_l2912_291209

def total_production : ℕ := 5650000
def first_supplier : ℕ := 1000000
def second_supplier : ℕ := first_supplier + 500000
def third_supplier : ℕ := first_supplier + second_supplier

theorem car_distribution (fourth_supplier fifth_supplier : ℕ) : 
  fourth_supplier = fifth_supplier ∧
  first_supplier + second_supplier + third_supplier + fourth_supplier + fifth_supplier = total_production →
  fourth_supplier = 325000 := by
sorry

end NUMINAMATH_CALUDE_car_distribution_l2912_291209


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2912_291247

/-- Represents the number of fire drill sites -/
def num_sites : ℕ := 3

/-- Represents the number of fire brigades -/
def num_brigades : ℕ := 4

/-- Represents the condition that each site must have at least one brigade -/
def min_brigade_per_site : ℕ := 1

/-- The number of ways to allocate fire brigades to sites -/
def allocation_schemes : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2912_291247


namespace NUMINAMATH_CALUDE_wood_square_weight_relation_l2912_291285

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation 
  (w1 w2 : WoodSquare)
  (uniform_density : True)  -- Represents the assumption of uniform density and thickness
  (h1 : w1.side_length = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side_length = 6) :
  w2.weight = 36 := by
  sorry

#check wood_square_weight_relation

end NUMINAMATH_CALUDE_wood_square_weight_relation_l2912_291285


namespace NUMINAMATH_CALUDE_clock_setback_radians_l2912_291260

theorem clock_setback_radians (minutes_per_revolution : ℝ) (radians_per_revolution : ℝ) 
  (setback_minutes : ℝ) : 
  minutes_per_revolution = 60 → 
  radians_per_revolution = 2 * Real.pi → 
  setback_minutes = 10 → 
  (setback_minutes / minutes_per_revolution) * radians_per_revolution = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_clock_setback_radians_l2912_291260


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l2912_291297

/-- The charge for each additional 1/5 mile in a taxi ride -/
def additional_fifth_mile_charge : ℚ := 0.40

/-- The initial charge for the first 1/5 mile -/
def initial_charge : ℚ := 3.00

/-- The total charge for an 8-mile ride -/
def total_charge_8_miles : ℚ := 18.60

/-- The length of the ride in miles -/
def ride_length : ℚ := 8

theorem taxi_charge_proof :
  initial_charge + (ride_length * 5 - 1) * additional_fifth_mile_charge = total_charge_8_miles :=
by sorry

end NUMINAMATH_CALUDE_taxi_charge_proof_l2912_291297


namespace NUMINAMATH_CALUDE_not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l2912_291267

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬ is_neighboring_root_equation 1 (-1) (-6) :=
sorry

/-- Theorem for the second equation -/
theorem neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 3) 1 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l2912_291267


namespace NUMINAMATH_CALUDE_k_gonal_number_formula_l2912_291258

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem: The formula for the n-th k-gonal number -/
theorem k_gonal_number_formula (n k : ℕ) (h1 : n ≥ 1) (h2 : k ≥ 3) :
  N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n :=
by sorry

end NUMINAMATH_CALUDE_k_gonal_number_formula_l2912_291258


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l2912_291250

theorem base_4_9_digit_difference :
  let n : ℕ := 1234
  let base_4_digits := (Nat.log n 4).succ
  let base_9_digits := (Nat.log n 9).succ
  base_4_digits = base_9_digits + 2 :=
by sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l2912_291250


namespace NUMINAMATH_CALUDE_sphere_to_cone_radius_l2912_291245

/-- The radius of a sphere that transforms into a cone with equal volume --/
theorem sphere_to_cone_radius (r : ℝ) (h : r = 3 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, 
    (4/3) * Real.pi * R^3 = 2 * Real.pi * r^3 ∧ 
    R = 3 * Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_to_cone_radius_l2912_291245


namespace NUMINAMATH_CALUDE_find_certain_number_l2912_291298

theorem find_certain_number (G N : ℕ) (h1 : G = 88) (h2 : N % G = 31) (h3 : 4521 % G = 33) : N = 4519 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l2912_291298


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2912_291277

/-- An arithmetic sequence where each term is not 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≠ 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_product (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2912_291277


namespace NUMINAMATH_CALUDE_prob_no_roots_l2912_291200

/-- A random variable following a normal distribution with mean 1 and variance s² -/
def normal_dist (s : ℝ) : Type := ℝ

/-- The probability density function of a normal distribution -/
noncomputable def pdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def cdf (s : ℝ) (x : ℝ) : ℝ := sorry

/-- The quadratic function f(x) = x² + 2x + ξ -/
def f (ξ : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + ξ

/-- The statement that f(x) has no roots -/
def no_roots (ξ : ℝ) : Prop := ∀ x, f ξ x ≠ 0

/-- The main theorem -/
theorem prob_no_roots (s : ℝ) (h : s > 0) : 
  (1 - cdf s 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_no_roots_l2912_291200


namespace NUMINAMATH_CALUDE_motorboat_trip_time_l2912_291253

theorem motorboat_trip_time (v_b : ℝ) (d : ℝ) (h1 : v_b > 0) (h2 : d > 0) : 
  let v_c := v_b / 3
  let t_no_current := 2 * d / v_b
  let v_down := v_b + v_c
  let v_up := v_b - v_c
  let t_actual := d / v_down + d / v_up
  t_no_current = 44 / 60 → t_actual = 49.5 / 60 := by
sorry

end NUMINAMATH_CALUDE_motorboat_trip_time_l2912_291253


namespace NUMINAMATH_CALUDE_basketball_contest_l2912_291296

/-- Calculates the total points scored in a basketball contest --/
def total_points (layups dunks free_throws three_pointers alley_oops half_court consecutive : ℕ) : ℕ :=
  layups + dunks + 2 * free_throws + 3 * three_pointers + 4 * alley_oops + 5 * half_court + consecutive

/-- Represents the basketball contest between Reggie and his brother --/
theorem basketball_contest :
  let reggie_points := total_points 4 2 3 2 1 1 2
  let brother_points := total_points 3 1 2 5 2 4 3
  brother_points - reggie_points = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_contest_l2912_291296


namespace NUMINAMATH_CALUDE_nineteen_customers_without_fish_l2912_291276

/-- Represents the fish market scenario --/
structure FishMarket where
  total_customers : ℕ
  tuna_count : ℕ
  tuna_weight : ℕ
  regular_customer_request : ℕ
  special_customer_30lb : ℕ
  special_customer_20lb : ℕ
  max_cuts_per_tuna : ℕ

/-- Calculates the number of customers who will go home without fish --/
def customers_without_fish (market : FishMarket) : ℕ :=
  let total_weight := market.tuna_count * market.tuna_weight
  let weight_for_30lb := market.special_customer_30lb * 30
  let weight_for_20lb := market.special_customer_20lb * 20
  let remaining_weight := total_weight - weight_for_30lb - weight_for_20lb
  let remaining_customers := remaining_weight / market.regular_customer_request
  let total_served := market.special_customer_30lb + market.special_customer_20lb + remaining_customers
  market.total_customers - total_served

/-- Theorem stating that 19 customers will go home without fish --/
theorem nineteen_customers_without_fish (market : FishMarket) 
  (h1 : market.total_customers = 100)
  (h2 : market.tuna_count = 10)
  (h3 : market.tuna_weight = 200)
  (h4 : market.regular_customer_request = 25)
  (h5 : market.special_customer_30lb = 10)
  (h6 : market.special_customer_20lb = 15)
  (h7 : market.max_cuts_per_tuna = 8) :
  customers_without_fish market = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_customers_without_fish_l2912_291276


namespace NUMINAMATH_CALUDE_complement_of_A_l2912_291235

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 1) * (x - 4) ≤ 0}

-- Theorem statement
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2912_291235


namespace NUMINAMATH_CALUDE_joan_gave_43_seashells_l2912_291214

/-- The number of seashells Joan initially found on the beach. -/
def initial_seashells : ℕ := 70

/-- The number of seashells Joan has left after giving some to Sam. -/
def remaining_seashells : ℕ := 27

/-- The number of seashells Joan gave to Sam. -/
def seashells_given_to_sam : ℕ := initial_seashells - remaining_seashells

/-- Theorem stating that Joan gave 43 seashells to Sam. -/
theorem joan_gave_43_seashells : seashells_given_to_sam = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_43_seashells_l2912_291214


namespace NUMINAMATH_CALUDE_john_lawyer_payment_l2912_291205

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontFee courtHours hourlyRate prepTimeFactor paperworkFee transportCost : ℕ) : ℕ :=
  let totalHours := courtHours * (1 + prepTimeFactor)
  let totalFee := upfrontFee + (totalHours * hourlyRate) + paperworkFee + transportCost
  totalFee / 2

theorem john_lawyer_payment :
  johnPayment 1000 50 100 2 500 300 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_john_lawyer_payment_l2912_291205


namespace NUMINAMATH_CALUDE_second_number_proof_l2912_291207

theorem second_number_proof (x : ℝ) : 
  let set1 := [10, 60, 35]
  let set2 := [20, 60, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 5 →
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2912_291207


namespace NUMINAMATH_CALUDE_total_ages_l2912_291221

/-- Given that Gabriel is 3 years younger than Frank and Frank is 10 years old,
    prove that the total of their ages is 17. -/
theorem total_ages (frank_age : ℕ) (gabriel_age : ℕ) : 
  frank_age = 10 → gabriel_age = frank_age - 3 → frank_age + gabriel_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_ages_l2912_291221


namespace NUMINAMATH_CALUDE_weekly_fat_intake_l2912_291256

def morning_rice : ℕ := 3
def afternoon_rice : ℕ := 2
def evening_rice : ℕ := 5
def fat_per_cup : ℕ := 10
def days_in_week : ℕ := 7

theorem weekly_fat_intake : 
  (morning_rice + afternoon_rice + evening_rice) * fat_per_cup * days_in_week = 700 := by
  sorry

end NUMINAMATH_CALUDE_weekly_fat_intake_l2912_291256


namespace NUMINAMATH_CALUDE_system_solution_l2912_291294

theorem system_solution (a₁ a₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x + y = c₁ ∧ a₂ * x + y = c₂ ∧ x = 5 ∧ y = 10) →
  (∃ (x y : ℝ), a₁ * x + 2 * y = a₁ - c₁ ∧ a₂ * x + 2 * y = a₂ - c₂ ∧ x = -4 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2912_291294


namespace NUMINAMATH_CALUDE_line_parameterization_l2912_291282

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f t, 20t - 10),
    prove that f t = 10t + 10 for all t. -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t, f t = 10 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2912_291282


namespace NUMINAMATH_CALUDE_total_working_days_l2912_291299

/-- Represents the commute options for a worker over a period of working days. -/
structure CommuteData where
  /-- Number of days the worker drove to work in the morning -/
  morning_drives : ℕ
  /-- Number of days the worker took the subway home in the afternoon -/
  afternoon_subways : ℕ
  /-- Total number of subway commutes (morning or afternoon) -/
  total_subway_commutes : ℕ

/-- Theorem stating that given the specific commute data, the total number of working days is 15 -/
theorem total_working_days (data : CommuteData) 
  (h1 : data.morning_drives = 12)
  (h2 : data.afternoon_subways = 20)
  (h3 : data.total_subway_commutes = 15) :
  data.morning_drives + (data.total_subway_commutes - data.morning_drives) = 15 := by
  sorry

#check total_working_days

end NUMINAMATH_CALUDE_total_working_days_l2912_291299


namespace NUMINAMATH_CALUDE_weeks_to_save_is_36_l2912_291255

/-- The number of weeks Nina needs to save to buy all items -/
def weeks_to_save : ℕ :=
let video_game_cost : ℚ := 50
let headset_cost : ℚ := 70
let gift_cost : ℚ := 30
let sales_tax_rate : ℚ := 12 / 100
let weekly_allowance : ℚ := 10
let initial_savings_rate : ℚ := 33 / 100
let later_savings_rate : ℚ := 50 / 100
let initial_savings_weeks : ℕ := 6

let total_cost_before_tax : ℚ := video_game_cost + headset_cost + gift_cost
let total_cost_with_tax : ℚ := total_cost_before_tax * (1 + sales_tax_rate)
let gift_cost_with_tax : ℚ := gift_cost * (1 + sales_tax_rate)

let initial_savings : ℚ := weekly_allowance * initial_savings_rate * initial_savings_weeks
let remaining_gift_cost : ℚ := gift_cost_with_tax - initial_savings
let weeks_for_gift : ℕ := (remaining_gift_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

let remaining_cost : ℚ := total_cost_with_tax - gift_cost_with_tax
let weeks_for_remaining : ℕ := (remaining_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

initial_savings_weeks + weeks_for_gift + weeks_for_remaining

theorem weeks_to_save_is_36 : weeks_to_save = 36 := by sorry

end NUMINAMATH_CALUDE_weeks_to_save_is_36_l2912_291255


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2912_291230

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  a^3 - ((a - 2) * a * (a + 2)) = 16 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2912_291230


namespace NUMINAMATH_CALUDE_max_k_for_sqrt_inequality_l2912_291201

theorem max_k_for_sqrt_inequality : 
  (∃ (k : ℝ), ∀ (l : ℝ), 
    (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ l) → 
    k ≥ l) ∧ 
  (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_sqrt_inequality_l2912_291201


namespace NUMINAMATH_CALUDE_price_adjustment_theorem_l2912_291257

theorem price_adjustment_theorem (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let first_increase := 1.20
  let second_increase := 1.10
  let third_increase := 1.15
  let discount := 0.95
  let tax := 1.07
  let final_price := original_price * first_increase * second_increase * third_increase * discount * tax
  let required_decrease := 0.351852
  final_price * (1 - required_decrease) = original_price := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_theorem_l2912_291257


namespace NUMINAMATH_CALUDE_infinitely_many_not_sum_of_seven_sixth_powers_l2912_291244

theorem infinitely_many_not_sum_of_seven_sixth_powers :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ a ∈ S, ∀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ, 
   a ≠ a₁^6 + a₂^6 + a₃^6 + a₄^6 + a₅^6 + a₆^6 + a₇^6) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_not_sum_of_seven_sixth_powers_l2912_291244


namespace NUMINAMATH_CALUDE_max_diagonal_sum_l2912_291220

/-- A rhombus with side length 5 -/
structure Rhombus where
  side_length : ℝ
  side_length_eq : side_length = 5

/-- The diagonals of the rhombus -/
structure RhombusDiagonals (r : Rhombus) where
  d1 : ℝ
  d2 : ℝ
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The sum of the diagonals -/
def diagonal_sum (r : Rhombus) (d : RhombusDiagonals r) : ℝ := d.d1 + d.d2

/-- The theorem stating the maximum sum of diagonals -/
theorem max_diagonal_sum (r : Rhombus) :
  ∃ (d : RhombusDiagonals r), ∀ (d' : RhombusDiagonals r), diagonal_sum r d ≥ diagonal_sum r d' ∧ diagonal_sum r d = 14 :=
sorry

end NUMINAMATH_CALUDE_max_diagonal_sum_l2912_291220


namespace NUMINAMATH_CALUDE_min_value_of_inverse_squares_l2912_291243

theorem min_value_of_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  (∃! (l : ℝ → ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) → 
    y = l x ∧ (∀ (x' y' : ℝ), y' = l x' → (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 > 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 > 0) ∨
    (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 < 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 < 0))) →
  (1 / a^2 + 1 / b^2 ≥ 9) ∧ (∃ (a' b' : ℝ), a' ≠ 0 ∧ b' ≠ 0 ∧ 1 / a'^2 + 1 / b'^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_inverse_squares_l2912_291243


namespace NUMINAMATH_CALUDE_expression_evaluation_l2912_291268

theorem expression_evaluation (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (-1/2 * b) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2912_291268


namespace NUMINAMATH_CALUDE_find_x_l2912_291278

theorem find_x : ∃ x : ℤ, (9873 + x = 13800) ∧ (x = 3927) := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2912_291278


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_axis_of_symmetry_example_l2912_291248

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is x = -b / (2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₀ : ℝ, x₀ = -b / (2 * a) ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 4x + 1 is the line x = 2 -/
theorem axis_of_symmetry_example :
  let f : ℝ → ℝ := λ x => -x^2 + 4*x + 1
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_axis_of_symmetry_example_l2912_291248


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2912_291271

/-- Given a triangle with side lengths 8, 2x+5, and 3x+2, and a perimeter of 40,
    the longest side of the triangle is 17. -/
theorem longest_side_of_triangle (x : ℝ) : 
  8 + (2*x + 5) + (3*x + 2) = 40 → 
  max 8 (max (2*x + 5) (3*x + 2)) = 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2912_291271


namespace NUMINAMATH_CALUDE_train_speed_l2912_291289

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 20) :
  (train_length + bridge_length) / crossing_time = 19.5 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2912_291289


namespace NUMINAMATH_CALUDE_shift_f_equals_g_l2912_291272

def f (x : ℝ) : ℝ := -x^2

def g (x : ℝ) : ℝ := -x^2 + 2

def vertical_shift (h : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => h x + k

theorem shift_f_equals_g : vertical_shift f 2 = g := by sorry

end NUMINAMATH_CALUDE_shift_f_equals_g_l2912_291272


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2912_291249

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2912_291249


namespace NUMINAMATH_CALUDE_students_in_both_events_l2912_291240

theorem students_in_both_events (total : ℕ) (volleyball : ℕ) (track_field : ℕ) (none : ℕ) :
  total = 45 →
  volleyball = 12 →
  track_field = 20 →
  none = 19 →
  volleyball + track_field - (total - none) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_both_events_l2912_291240


namespace NUMINAMATH_CALUDE_particle_speed_l2912_291279

/-- A particle moves so that its position at time t is (3t + 5, 6t - 11).
    This function represents the particle's position vector at time t. -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 6 * t - 11)

/-- The speed of the particle is the magnitude of the change in position vector
    per unit time interval. -/
theorem particle_speed : 
  let v := particle_position 1 - particle_position 0
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_l2912_291279


namespace NUMINAMATH_CALUDE_locus_of_point_c_l2912_291273

/-- Given a right triangle ABC with ∠C = 90°, where A is on the positive x-axis and B is on the positive y-axis,
    prove that the locus of point C is described by the equation y = (b/a)x, where ab/c ≤ x ≤ a. -/
theorem locus_of_point_c (a b c : ℝ) (A B C : ℝ × ℝ) :
  a > 0 → b > 0 →
  c^2 = a^2 + b^2 →
  A.1 > 0 → A.2 = 0 →
  B.1 = 0 → B.2 > 0 →
  C.1^2 + C.2^2 = a^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = b^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 →
  ∃ (x : ℝ), a*b/c ≤ x ∧ x ≤ a ∧ C = (x, b/a * x) :=
sorry


end NUMINAMATH_CALUDE_locus_of_point_c_l2912_291273


namespace NUMINAMATH_CALUDE_petya_marking_strategy_l2912_291227

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- The minimum number of cells needed to be marked to uniquely determine 
    the position of a rectangle on a board -/
def min_marked_cells (b : Board) (r : Rectangle) : ℕ := sorry

/-- The main theorem stating the minimum number of cells Petya needs to mark -/
theorem petya_marking_strategy (b : Board) (r : Rectangle) : 
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 := by sorry

end NUMINAMATH_CALUDE_petya_marking_strategy_l2912_291227


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2912_291274

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 135) 
  (h_ratio : diagonal_ratio = 5 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ d1 * d2 / 2 = area ∧ d1 = 15 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2912_291274


namespace NUMINAMATH_CALUDE_complement_A_eq_l2912_291270

/-- The universal set U -/
def U : Set Int := {-2, -1, 1, 3, 5}

/-- The set A -/
def A : Set Int := {-1, 3}

/-- The complement of A with respect to U -/
def complement_A : Set Int := {x | x ∈ U ∧ x ∉ A}

theorem complement_A_eq : complement_A = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_eq_l2912_291270


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2912_291284

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 1) * x + 1 < 0 ↔ 
    (a = 0 ∧ x > 1) ∨
    (a < 0 ∧ (x < 1/a ∨ x > 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
    (a > 1 ∧ 1/a < x ∧ x < 1) ∨
    (a ≠ 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2912_291284


namespace NUMINAMATH_CALUDE_rectangle_strip_count_l2912_291222

theorem rectangle_strip_count 
  (outer_perimeter : ℕ) 
  (hole_perimeter : ℕ) 
  (horizontal_strips : ℕ) : 
  outer_perimeter = 50 → 
  hole_perimeter = 32 → 
  horizontal_strips = 20 → 
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_strip_count_l2912_291222


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2912_291217

theorem sum_of_squares_and_square_of_sum : (4 + 8)^2 + (4^2 + 8^2) = 224 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2912_291217


namespace NUMINAMATH_CALUDE_jorges_clay_rich_soil_fraction_l2912_291251

theorem jorges_clay_rich_soil_fraction (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_rich_soil_yield : ℝ) (total_yield : ℝ) 
  (h1 : total_land = 60)
  (h2 : good_soil_yield = 400)
  (h3 : clay_rich_soil_yield = good_soil_yield / 2)
  (h4 : total_yield = 20000) :
  let clay_rich_fraction := (total_land * good_soil_yield - total_yield) / 
    (total_land * (good_soil_yield - clay_rich_soil_yield))
  clay_rich_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jorges_clay_rich_soil_fraction_l2912_291251


namespace NUMINAMATH_CALUDE_initial_books_eq_sold_plus_unsold_l2912_291239

/-- The number of books Ali had initially --/
def initial_books : ℕ := sorry

/-- The number of books Ali sold on Monday --/
def monday_sales : ℕ := 60

/-- The number of books Ali sold on Tuesday --/
def tuesday_sales : ℕ := 10

/-- The number of books Ali sold on Wednesday --/
def wednesday_sales : ℕ := 20

/-- The number of books Ali sold on Thursday --/
def thursday_sales : ℕ := 44

/-- The number of books Ali sold on Friday --/
def friday_sales : ℕ := 66

/-- The number of books not sold --/
def unsold_books : ℕ := 600

/-- Theorem stating that the initial number of books is equal to the sum of books sold on each day plus the number of books not sold --/
theorem initial_books_eq_sold_plus_unsold :
  initial_books = monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + unsold_books := by
  sorry

end NUMINAMATH_CALUDE_initial_books_eq_sold_plus_unsold_l2912_291239


namespace NUMINAMATH_CALUDE_breadth_is_five_l2912_291261

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 15 * breadth
  length_diff : length = breadth + 10

/-- The breadth of a rectangular plot with given properties is 5 meters -/
theorem breadth_is_five (plot : RectangularPlot) : plot.breadth = 5 := by
  sorry

end NUMINAMATH_CALUDE_breadth_is_five_l2912_291261


namespace NUMINAMATH_CALUDE_arrange_five_from_ten_eq_30240_l2912_291280

/-- The number of ways to arrange 5 distinct numbers from a set of 10 numbers -/
def arrange_five_from_ten : ℕ := 10 * 9 * 8 * 7 * 6

/-- Theorem stating that arranging 5 distinct numbers from a set of 10 numbers results in 30240 possibilities -/
theorem arrange_five_from_ten_eq_30240 : arrange_five_from_ten = 30240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_from_ten_eq_30240_l2912_291280


namespace NUMINAMATH_CALUDE_valid_numbers_l2912_291203

def is_valid_number (n : ℕ) : Prop :=
  Odd n ∧
  ∃ (a b : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    (∃ (k : ℕ), n = 10^k * a + b) ∧
    n = 149 * b

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 745 ∨ n = 3725 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2912_291203


namespace NUMINAMATH_CALUDE_existence_of_g_l2912_291254

open Set
open Function
open ContinuousOn

theorem existence_of_g (a b : ℝ) (f : ℝ → ℝ) 
  (h_f_cont : ContinuousOn f (Icc a b))
  (h_f_deriv : DifferentiableOn ℝ f (Icc a b))
  (h_f_zero : ∀ x ∈ Icc a b, f x = 0 → deriv f x ≠ 0) :
  ∃ g : ℝ → ℝ, 
    ContinuousOn g (Icc a b) ∧ 
    DifferentiableOn ℝ g (Icc a b) ∧
    ∀ x ∈ Icc a b, f x * deriv g x > deriv f x * g x :=
sorry

end NUMINAMATH_CALUDE_existence_of_g_l2912_291254
