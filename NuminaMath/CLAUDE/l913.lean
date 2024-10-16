import Mathlib

namespace NUMINAMATH_CALUDE_average_of_six_numbers_l913_91321

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l913_91321


namespace NUMINAMATH_CALUDE_lcm_of_4_6_9_l913_91320

theorem lcm_of_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_9_l913_91320


namespace NUMINAMATH_CALUDE_xy_and_expression_values_l913_91356

theorem xy_and_expression_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : (x - 2)*(y + 1) = 2) : 
  x*y = 1 ∧ (x^2 - 2)*(2*y^2 - 1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_expression_values_l913_91356


namespace NUMINAMATH_CALUDE_cubic_equation_solution_sum_l913_91366

theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 5*r^2 + 6*r = 9 →
  s^3 - 5*s^2 + 6*s = 9 →
  t^3 - 5*t^2 + 6*t = 9 →
  r*s/t + s*t/r + t*r/s = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_sum_l913_91366


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_five_l913_91324

theorem no_real_roots_x_squared_plus_five :
  ∀ x : ℝ, x^2 + 5 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_five_l913_91324


namespace NUMINAMATH_CALUDE_all_judgments_correct_l913_91300

theorem all_judgments_correct (a b c : ℕ) (ha : a = 2^22) (hb : b = 3^11) (hc : c = 12^9) :
  (a > b) ∧ (a * b > c) ∧ (b < c) := by
  sorry

end NUMINAMATH_CALUDE_all_judgments_correct_l913_91300


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l913_91384

theorem complex_subtraction_simplification :
  (5 - 7 * Complex.I) - (3 - 2 * Complex.I) = 2 - 5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l913_91384


namespace NUMINAMATH_CALUDE_unique_projection_l913_91329

def vector_projection (a b s p : ℝ × ℝ) : Prop :=
  let line_dir := (b.1 - a.1, b.2 - a.2)
  let shifted_line (t : ℝ) := (a.1 + s.1 + t * line_dir.1, a.2 + s.2 + t * line_dir.2)
  ∃ t : ℝ, 
    p = shifted_line t ∧ 
    line_dir.1 * (p.1 - (a.1 + s.1)) + line_dir.2 * (p.2 - (a.2 + s.2)) = 0

theorem unique_projection :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (-1, 4)
  let s : ℝ × ℝ := (1, 1)
  let p : ℝ × ℝ := (16/13, 41/26)
  vector_projection a b s p ∧ 
  ∀ q : ℝ × ℝ, vector_projection a b s q → q = p := by sorry

end NUMINAMATH_CALUDE_unique_projection_l913_91329


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l913_91350

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) :
  let new_length := L * (1 + x / 100)
  let new_breadth := B * 0.9
  let new_area := new_length * new_breadth
  let original_area := L * B
  new_area = original_area * 1.035 → x = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l913_91350


namespace NUMINAMATH_CALUDE_arrangement_count_is_120_l913_91385

/-- The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions -/
def arrangement_count : ℕ := 5 * 4 * 3 * 2 * 1

/-- Theorem: The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions is 120 -/
theorem arrangement_count_is_120 : arrangement_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_120_l913_91385


namespace NUMINAMATH_CALUDE_amys_flash_drive_storage_l913_91376

/-- Calculates the total storage space used on Amy's flash drive -/
def total_storage_space (music_files : ℝ) (music_size : ℝ) (video_files : ℝ) (video_size : ℝ) (picture_files : ℝ) (picture_size : ℝ) : ℝ :=
  music_files * music_size + video_files * video_size + picture_files * picture_size

/-- Theorem: The total storage space used on Amy's flash drive is 1116 MB -/
theorem amys_flash_drive_storage :
  total_storage_space 4 5 21 50 23 2 = 1116 := by
  sorry

end NUMINAMATH_CALUDE_amys_flash_drive_storage_l913_91376


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l913_91333

/-- The line equation (m-1)x + (2m-1)y = m-5 always passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l913_91333


namespace NUMINAMATH_CALUDE_min_real_roots_2010_l913_91345

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := { p : Polynomial ℝ // p.degree = 2010 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The theorem statement -/
theorem min_real_roots_2010 (p : RealPolynomial2010) 
  (h : distinctAbsValues p = 1010) : 
  realRootCount p ≥ 10 := sorry

end NUMINAMATH_CALUDE_min_real_roots_2010_l913_91345


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l913_91357

/-- Given a square with diagonal length 20, prove that its perimeter is 40√2 -/
theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 20) :
  let s := Real.sqrt (d^2 / 2)
  4 * s = 40 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l913_91357


namespace NUMINAMATH_CALUDE_smallest_cut_length_l913_91387

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 12 ∧ (12 - x) + (20 - x) ≤ (24 - x) →
  x ≥ 8 ∧ ∀ y : ℕ, y > 0 ∧ y < x → (12 - y) + (20 - y) > (24 - y) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l913_91387


namespace NUMINAMATH_CALUDE_average_page_count_l913_91371

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 ∧ g2 = g3 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 →
  n = 15 ∧ g1 = 5 ∧ p1 = 2 ∧ p2 = 3 ∧ p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_page_count_l913_91371


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l913_91303

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 7) = k →
  (3 * x₂ - 4) / (y₂ + 7) = k →
  x₁ = 3 →
  y₁ = 5 →
  y₂ = 20 →
  x₂ = 5.0833 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l913_91303


namespace NUMINAMATH_CALUDE_power_sum_seven_l913_91389

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^7 + b^7 = 29 -/
theorem power_sum_seven (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h6 : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^7 + b^7 = 29 := by
sorry

end NUMINAMATH_CALUDE_power_sum_seven_l913_91389


namespace NUMINAMATH_CALUDE_dogs_with_tags_l913_91352

theorem dogs_with_tags (total : ℕ) (with_flea_collars : ℕ) (with_both : ℕ) (with_neither : ℕ) : 
  total = 80 → 
  with_flea_collars = 40 → 
  with_both = 6 → 
  with_neither = 1 → 
  total - with_flea_collars + with_both - with_neither = 45 := by
sorry

end NUMINAMATH_CALUDE_dogs_with_tags_l913_91352


namespace NUMINAMATH_CALUDE_f_properties_l913_91365

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_properties :
  (f 2 = 4) ∧
  (f (1/2) = 1/4) ∧
  (f (f (-1)) = 1) ∧
  (∃ a : ℝ, f a = 3 ∧ (a = 1 ∨ a = Real.sqrt 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l913_91365


namespace NUMINAMATH_CALUDE_hidden_primes_average_l913_91348

/-- Given three cards with numbers on both sides, this theorem proves that
    the average of the hidden prime numbers is 46/3, given the conditions
    specified in the problem. -/
theorem hidden_primes_average (card1_visible card2_visible card3_visible : ℕ)
  (card1_hidden card2_hidden card3_hidden : ℕ)
  (h1 : card1_visible = 68)
  (h2 : card2_visible = 39)
  (h3 : card3_visible = 57)
  (h4 : Nat.Prime card1_hidden)
  (h5 : Nat.Prime card2_hidden)
  (h6 : Nat.Prime card3_hidden)
  (h7 : card1_visible + card1_hidden = card2_visible + card2_hidden)
  (h8 : card2_visible + card2_hidden = card3_visible + card3_hidden) :
  (card1_hidden + card2_hidden + card3_hidden : ℚ) / 3 = 46 / 3 := by
  sorry

#eval (46 : ℚ) / 3

end NUMINAMATH_CALUDE_hidden_primes_average_l913_91348


namespace NUMINAMATH_CALUDE_cos_sin_2theta_l913_91347

theorem cos_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) :
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_2theta_l913_91347


namespace NUMINAMATH_CALUDE_paint_replacement_fractions_l913_91364

/-- Represents the fraction of paint replaced -/
def fraction_replaced (initial_intensity final_intensity new_intensity : ℚ) : ℚ :=
  (initial_intensity - final_intensity) / (initial_intensity - new_intensity)

theorem paint_replacement_fractions :
  let red_initial := (50 : ℚ) / 100
  let blue_initial := (60 : ℚ) / 100
  let red_new := (35 : ℚ) / 100
  let blue_new := (45 : ℚ) / 100
  let red_final := (45 : ℚ) / 100
  let blue_final := (55 : ℚ) / 100
  (fraction_replaced red_initial red_final red_new = 1/3) ∧
  (fraction_replaced blue_initial blue_final blue_new = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_paint_replacement_fractions_l913_91364


namespace NUMINAMATH_CALUDE_unique_prime_solution_l913_91383

theorem unique_prime_solution : ∃! (p : ℕ), Prime p ∧ (p^4 + 2*p^3 + 4*p^2 + 2*p + 1)^5 = 418195493 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l913_91383


namespace NUMINAMATH_CALUDE_marble_distribution_l913_91353

theorem marble_distribution (total_marbles : ℕ) (ratio_a ratio_b : ℕ) (given_marbles : ℕ) : 
  total_marbles = 36 →
  ratio_a = 4 →
  ratio_b = 5 →
  given_marbles = 2 →
  (ratio_b * (total_marbles / (ratio_a + ratio_b))) - given_marbles = 18 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l913_91353


namespace NUMINAMATH_CALUDE_max_servings_is_16_l913_91310

/-- Represents the number of servings that can be made from a given ingredient --/
def servings_from_ingredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Represents the available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (min (servings_from_ingredient available.bananas recipe.bananas)
         (servings_from_ingredient available.yogurt recipe.yogurt))
    (min (servings_from_ingredient available.honey recipe.honey)
         (servings_from_ingredient available.strawberries recipe.strawberries))

theorem max_servings_is_16 (recipe : Recipe) (available : Available) :
  recipe.bananas = 3 ∧ recipe.yogurt = 1 ∧ recipe.honey = 2 ∧ recipe.strawberries = 2 ∧
  available.bananas = 12 ∧ available.yogurt = 6 ∧ available.honey = 16 ∧ available.strawberries = 8 →
  max_servings recipe available = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_16_l913_91310


namespace NUMINAMATH_CALUDE_exists_farther_point_l913_91351

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of the box -/
inductive SurfacePoint (b : Box)
  | front (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | back (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | left (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | right (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | top (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | bottom (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b

/-- The distance between two points on the surface of the box -/
def surfaceDistance (b : Box) (p q : SurfacePoint b) : ℝ := sorry

/-- The opposite corner of a given corner -/
def oppositeCorner (b : Box) (p : SurfacePoint b) : SurfacePoint b := sorry

/-- Theorem: There exists a point on the surface farther from a corner than the opposite corner -/
theorem exists_farther_point (b : Box) :
  ∃ (corner : SurfacePoint b) (p : SurfacePoint b),
    surfaceDistance b corner p > surfaceDistance b corner (oppositeCorner b corner) := by sorry

end NUMINAMATH_CALUDE_exists_farther_point_l913_91351


namespace NUMINAMATH_CALUDE_rectangle_side_length_l913_91308

theorem rectangle_side_length (perimeter width : ℝ) (h1 : perimeter = 40) (h2 : width = 8) :
  perimeter / 2 - width = 12 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l913_91308


namespace NUMINAMATH_CALUDE_jam_jar_max_theorem_l913_91381

/-- Represents the initial state of jam jars --/
structure JamJars :=
  (carlson_weight : ℕ)
  (baby_weight : ℕ)
  (carlson_min_jar : ℕ)

/-- Conditions for the jam jar problem --/
def valid_jam_jars (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.carlson_min_jar = 8 * (j.baby_weight + j.carlson_min_jar)

/-- The maximum number of jars Carlson could have initially --/
def max_carlson_jars : ℕ := 23

/-- Theorem stating the maximum number of jars Carlson could have initially --/
theorem jam_jar_max_theorem (j : JamJars) (h : valid_jam_jars j) :
  (j.carlson_weight / j.carlson_min_jar : ℚ) ≤ max_carlson_jars :=
sorry

end NUMINAMATH_CALUDE_jam_jar_max_theorem_l913_91381


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l913_91306

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → 1015 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l913_91306


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l913_91331

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / (x - 1) = 2 / (x - 2)) ∧ (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l913_91331


namespace NUMINAMATH_CALUDE_magnitude_v_l913_91392

theorem magnitude_v (u v : ℂ) (h1 : u * v = 16 - 30 * I) (h2 : Complex.abs u = 2) : 
  Complex.abs v = 17 := by
sorry

end NUMINAMATH_CALUDE_magnitude_v_l913_91392


namespace NUMINAMATH_CALUDE_f_monotone_and_F_lower_bound_l913_91344

noncomputable section

variables (m : ℝ) (x x₀ : ℝ)

def f (x : ℝ) : ℝ := x * Real.exp x - m * x

def F (x : ℝ) : ℝ := f m x - m * Real.log x

theorem f_monotone_and_F_lower_bound (hm : m < -Real.exp (-2)) 
  (h_crit : deriv (F m) x₀ = 0) (h_pos : F m x₀ > 0) :
  (∀ x y, x < y → f m x < f m y) ∧ F m x₀ > -2 * x₀^3 + 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_and_F_lower_bound_l913_91344


namespace NUMINAMATH_CALUDE_playground_children_count_l913_91334

/-- The number of boys on the playground at recess -/
def num_boys : ℕ := 27

/-- The number of girls on the playground at recess -/
def num_girls : ℕ := 35

/-- The total number of children on the playground at recess -/
def total_children : ℕ := num_boys + num_girls

theorem playground_children_count : total_children = 62 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l913_91334


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l913_91340

theorem product_of_positive_real_solutions : ∃ (S : Finset (ℂ)),
  (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧
  (∀ z : ℂ, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧
  (S.prod id = 8) :=
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l913_91340


namespace NUMINAMATH_CALUDE_first_division_divisor_l913_91311

theorem first_division_divisor
  (x : ℕ+) -- x is a positive integer
  (y : ℕ) -- y is a natural number (quotient)
  (d : ℕ) -- d is the divisor we're looking for
  (h1 : ∃ q : ℕ, x = d * y + 3) -- x divided by d gives quotient y and remainder 3
  (h2 : ∃ q : ℕ, 2 * x = 7 * (3 * y) + 1) -- 2x divided by 7 gives quotient 3y and remainder 1
  (h3 : 11 * y - x = 2) -- Given equation
  : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_division_divisor_l913_91311


namespace NUMINAMATH_CALUDE_percent_division_multiplication_equality_l913_91318

theorem percent_division_multiplication_equality : 
  (30 / 100 : ℚ) / (1 + 2 / 5) * (1 / 3 + 1 / 7) = 5 / 49 := by sorry

end NUMINAMATH_CALUDE_percent_division_multiplication_equality_l913_91318


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l913_91305

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 1 ∧
  n % 13 = 2 ∧
  n % 7 = 2 ∧
  n % 3 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l913_91305


namespace NUMINAMATH_CALUDE_oocyte_characteristics_l913_91355

/-- Represents the hair length trait in rabbits -/
inductive HairTrait
  | Long
  | Short

/-- Represents the phase of meiosis -/
inductive MeioticPhase
  | First
  | Second

/-- Represents a heterozygous rabbit with long hair trait dominant over short hair trait -/
structure HeterozygousRabbit where
  dominantTrait : HairTrait
  recessiveTrait : HairTrait
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  allelesSeperationPhase : MeioticPhase

/-- Main theorem about the characteristics of oocytes in a heterozygous rabbit -/
theorem oocyte_characteristics (rabbit : HeterozygousRabbit)
  (h1 : rabbit.dominantTrait = HairTrait.Long)
  (h2 : rabbit.recessiveTrait = HairTrait.Short)
  (h3 : rabbit.totalGenes = 20)
  (h4 : rabbit.genesPerOocyte = 4)
  (h5 : rabbit.nucleotideTypes = 4)
  (h6 : rabbit.allelesSeperationPhase = MeioticPhase.First) :
  let maxShortHairOocytes := rabbit.totalGenes / rabbit.genesPerOocyte / 2
  maxShortHairOocytes = 5 ∧
  rabbit.nucleotideTypes = 4 ∧
  rabbit.allelesSeperationPhase = MeioticPhase.First :=
by sorry

end NUMINAMATH_CALUDE_oocyte_characteristics_l913_91355


namespace NUMINAMATH_CALUDE_kelvins_classes_l913_91302

theorem kelvins_classes (grant_vacations kelvin_classes : ℕ) 
  (h1 : grant_vacations = 4 * kelvin_classes)
  (h2 : grant_vacations + kelvin_classes = 450) :
  kelvin_classes = 90 := by
  sorry

end NUMINAMATH_CALUDE_kelvins_classes_l913_91302


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l913_91395

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ -(Real.sqrt 3) * x^2
  let slope : ℝ := f' 1
  let angle_with_neg_x : ℝ := Real.arctan (Real.sqrt 3)
  let angle_with_pos_x : ℝ := π - angle_with_neg_x
  angle_with_pos_x = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l913_91395


namespace NUMINAMATH_CALUDE_probability_two_ones_eight_dice_l913_91342

/-- The probability of exactly two dice showing a 1 when rolling eight standard 6-sided dice -/
def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of exactly two ones when rolling eight 6-sided dice -/
theorem probability_two_ones_eight_dice :
  probability_two_ones 8 2 (1/6) = 28 * (1/6)^2 * (5/6)^6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_ones_eight_dice_l913_91342


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l913_91394

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l913_91394


namespace NUMINAMATH_CALUDE_hyperbola_properties_l913_91319

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def distance_to_asymptote (h : Hyperbola) (l : Line) : ℝ := sorry

def standard_equation (h : Hyperbola) : Prop :=
  h.a = 1 ∧ h.b = 2

def slope_ratio (h : Hyperbola) (l : Line) : ℝ := sorry

def fixed_point_exists (h : Hyperbola) : Prop :=
  ∃ (G : Point), G.x = 1 ∧ G.y = 0 ∧
  ∀ (l : Line), slope_ratio h l = -1/3 →
  ∃ (H : Point), (H.x - G.x)^2 + (H.y - G.y)^2 = 1

theorem hyperbola_properties (h : Hyperbola) 
  (asymptote : Line)
  (h_asymptote : asymptote.m = 2 ∧ asymptote.c = 0)
  (h_distance : distance_to_asymptote h asymptote = 2) :
  standard_equation h ∧ fixed_point_exists h := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l913_91319


namespace NUMINAMATH_CALUDE_ratio_of_a_to_b_l913_91391

theorem ratio_of_a_to_b (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 4) * B) : A / B = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_b_l913_91391


namespace NUMINAMATH_CALUDE_shaded_area_of_perpendicular_diameters_l913_91323

theorem shaded_area_of_perpendicular_diameters (r : ℝ) (h : r = 4) : 
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let right_triangle_area := r^2 / 2
  2 * right_triangle_area + 2 * quarter_circle_area = 16 + 8 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_perpendicular_diameters_l913_91323


namespace NUMINAMATH_CALUDE_distance_between_intersecting_circles_l913_91397

/-- The distance between the centers of two intersecting circles -/
def distance_between_centers (a : ℝ) : Set ℝ :=
  {a / 6 * (3 + Real.sqrt 3), a / 6 * (3 - Real.sqrt 3)}

/-- Represents two intersecting circles with a common chord -/
structure IntersectingCircles (a : ℝ) where
  /-- The common chord length -/
  chord_length : ℝ
  /-- The chord is a side of a regular inscribed triangle in one circle -/
  is_triangle_side : Bool
  /-- The chord is a side of an inscribed square in the other circle -/
  is_square_side : Bool
  /-- The chord length is positive -/
  chord_positive : chord_length > 0
  /-- The chord length is equal to a -/
  chord_eq_a : chord_length = a
  /-- One circle has the chord as a triangle side, the other as a square side -/
  different_inscriptions : is_triangle_side ≠ is_square_side

/-- Theorem stating the distance between centers of intersecting circles -/
theorem distance_between_intersecting_circles (a : ℝ) (circles : IntersectingCircles a) :
  ∃ d ∈ distance_between_centers a,
    d = (circles.chord_length / 6) * (3 + Real.sqrt 3) ∨
    d = (circles.chord_length / 6) * (3 - Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_distance_between_intersecting_circles_l913_91397


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l913_91386

-- Define the sets P and Q
def P : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def Q : Set ℝ := {x | (1 + x) / (x - 3) ≤ 0}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l913_91386


namespace NUMINAMATH_CALUDE_candy_distribution_l913_91360

theorem candy_distribution (total_candy : Nat) (num_friends : Nat) : 
  total_candy = 379 → num_friends = 6 → 
  ∃ (equal_distribution : Nat), 
    equal_distribution ≤ total_candy ∧ 
    equal_distribution.mod num_friends = 0 ∧
    ∀ n : Nat, n ≤ total_candy ∧ n.mod num_friends = 0 → n ≤ equal_distribution := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l913_91360


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l913_91346

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l913_91346


namespace NUMINAMATH_CALUDE_divisor_problem_l913_91309

theorem divisor_problem (x k m y : ℤ) 
  (h1 : x = 62 * k + 7)
  (h2 : x + 11 = y * m + 18)
  (h3 : y > 18)
  (h4 : 62 % y = 0) :
  y = 31 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l913_91309


namespace NUMINAMATH_CALUDE_marks_fruit_purchase_l913_91380

/-- The total cost of Mark's fruit purchase --/
def total_cost (tomato_price apple_price orange_price : ℝ)
                (tomato_weight apple_weight orange_weight : ℝ)
                (apple_discount : ℝ) : ℝ :=
  tomato_price * tomato_weight +
  apple_price * apple_weight * (1 - apple_discount) +
  orange_price * orange_weight

/-- Theorem stating the total cost of Mark's fruit purchase --/
theorem marks_fruit_purchase :
  total_cost 4.50 3.25 2.75 3 7 4 0.1 = 44.975 := by
  sorry

#eval total_cost 4.50 3.25 2.75 3 7 4 0.1

end NUMINAMATH_CALUDE_marks_fruit_purchase_l913_91380


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_part1_simplify_and_evaluate_part2_l913_91304

-- Part 1
theorem simplify_and_evaluate_part1 :
  ∀ a : ℝ, 3*a*(a^2 - 2*a + 1) - 2*a^2*(a - 3) = a^3 + 3*a ∧
  3*2*(2^2 - 2*2 + 1) - 2*2^2*(2 - 3) = 14 :=
sorry

-- Part 2
theorem simplify_and_evaluate_part2 :
  ∀ x : ℝ, (x - 4)*(x - 2) - (x - 1)*(x + 3) = -8*x + 11 ∧
  ((-5/2) - 4)*((-5/2) - 2) - ((-5/2) - 1)*((-5/2) + 3) = 31 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_part1_simplify_and_evaluate_part2_l913_91304


namespace NUMINAMATH_CALUDE_min_value_theorem_l913_91370

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 4) :
  ∃ (m : ℝ), (3 * a + 1 ≥ m) ∧ (∀ x, 8 * x^2 + 6 * x + 2 = 4 → 3 * x + 1 ≥ m) ∧ (m = -2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l913_91370


namespace NUMINAMATH_CALUDE_constant_term_expansion_l913_91315

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (2*x - 1/(2*x))^6
  ∃ (a b c d e f g : ℝ), expansion = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (-20) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l913_91315


namespace NUMINAMATH_CALUDE_min_value_x_plus_inverse_y_l913_91372

theorem min_value_x_plus_inverse_y (x y : ℝ) (h1 : x ≥ 3) (h2 : x - y = 1) :
  ∃ m : ℝ, m = 7/2 ∧ ∀ z : ℝ, z ≥ 3 → ∀ w : ℝ, z - w = 1 → z + 1/w ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_inverse_y_l913_91372


namespace NUMINAMATH_CALUDE_find_number_l913_91336

theorem find_number (n x : ℝ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l913_91336


namespace NUMINAMATH_CALUDE_combined_mean_l913_91399

theorem combined_mean (set1_count : Nat) (set1_mean : ℝ) (set2_count : Nat) (set2_mean : ℝ) 
  (h1 : set1_count = 5)
  (h2 : set1_mean = 13)
  (h3 : set2_count = 6)
  (h4 : set2_mean = 24) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_mean_l913_91399


namespace NUMINAMATH_CALUDE_arts_students_count_l913_91327

/-- Represents the number of arts students in the college -/
def arts_students : ℕ := sorry

/-- Represents the number of local arts students -/
def local_arts_students : ℕ := sorry

/-- Represents the number of local science students -/
def local_science_students : ℕ := 25

/-- Represents the number of local commerce students -/
def local_commerce_students : ℕ := 102

/-- Represents the total number of local students -/
def total_local_students : ℕ := 327

/-- Theorem stating that the number of arts students is 400 -/
theorem arts_students_count :
  (local_arts_students = arts_students / 2) ∧
  (local_arts_students + local_science_students + local_commerce_students = total_local_students) →
  arts_students = 400 :=
by sorry

end NUMINAMATH_CALUDE_arts_students_count_l913_91327


namespace NUMINAMATH_CALUDE_fraction_equality_l913_91377

theorem fraction_equality (a b c d : ℝ) (h1 : b ≠ c) 
  (h2 : (a * c - b^2) / (a - 2*b + c) = (b * d - c^2) / (b - 2*c + d)) : 
  (a * c - b^2) / (a - 2*b + c) = (a * d - b * c) / (a - b - c + d) ∧ 
  (b * d - c^2) / (b - 2*c + d) = (a * d - b * c) / (a - b - c + d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l913_91377


namespace NUMINAMATH_CALUDE_square_function_difference_l913_91373

/-- For f(x) = x^2, prove that f(x) - f(x-1) = 2x - 1 for all real x -/
theorem square_function_difference (x : ℝ) : x^2 - (x-1)^2 = 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_function_difference_l913_91373


namespace NUMINAMATH_CALUDE_two_xy_equals_seven_l913_91317

theorem two_xy_equals_seven (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (2 : ℝ)^(x+y) = 64)
  (h2 : (9 : ℝ)^(x+y) / (3 : ℝ)^(4*y) = 243) :
  2 * x * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_xy_equals_seven_l913_91317


namespace NUMINAMATH_CALUDE_wire_bending_l913_91362

theorem wire_bending (r : ℝ) (h : r = 56) : 
  let circle_circumference := 2 * Real.pi * r
  let square_side := circle_circumference / 4
  let square_area := square_side * square_side
  square_area = 784 * Real.pi^2 := by
sorry

end NUMINAMATH_CALUDE_wire_bending_l913_91362


namespace NUMINAMATH_CALUDE_extreme_value_derivative_condition_l913_91396

open Real

theorem extreme_value_derivative_condition (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀ + ε ∨ f x ≥ f x₀ - ε) →
  (deriv f) x₀ = 0 ∧
  ∃ g : ℝ → ℝ, (deriv g) 0 = 0 ∧ ¬(∀ ε > 0, ∃ δ > 0, ∀ x, |x - 0| < δ → g x ≤ g 0 + ε ∨ g x ≥ g 0 - ε) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_condition_l913_91396


namespace NUMINAMATH_CALUDE_largest_possible_median_l913_91338

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 3, 2, 5}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem largest_possible_median (x : ℤ) :
  ∃ m : ℤ, is_median m (number_set x) ∧ ∀ n : ℤ, is_median n (number_set x) → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_largest_possible_median_l913_91338


namespace NUMINAMATH_CALUDE_cookie_batches_l913_91316

/-- The number of batches of cookies made from one bag of chocolate chips -/
def num_batches (chips_per_cookie : ℕ) (chips_per_bag : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  chips_per_bag / (chips_per_cookie * cookies_per_batch)

/-- Theorem: The number of batches of cookies made from one bag of chocolate chips is 3 -/
theorem cookie_batches :
  num_batches 9 81 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_batches_l913_91316


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l913_91326

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x)^2 + (y + 1/y)^2 ≥ 25/2 ∧
  ((x + 1/x)^2 + (y + 1/y)^2 = 25/2 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l913_91326


namespace NUMINAMATH_CALUDE_exists_fixed_point_l913_91358

variable {X : Type u}
variable (μ : Set X → Set X)

axiom μ_union_disjoint {A B : Set X} (h : Disjoint A B) : μ (A ∪ B) = μ A ∪ μ B

theorem exists_fixed_point : ∃ F : Set X, μ F = F := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l913_91358


namespace NUMINAMATH_CALUDE_peters_birdseed_for_week_l913_91375

/-- Calculate the total grams of birdseed Peter needs to buy for a week -/
theorem peters_birdseed_for_week (
  parakeet_count : ℕ)
  (parakeet_consumption : ℕ)
  (parrot_count : ℕ)
  (parrot_consumption : ℕ)
  (finch_count : ℕ)
  (canary_count : ℕ)
  (canary_consumption : ℕ)
  (african_grey_count : ℕ)
  (african_grey_consumption : ℕ)
  (toucan_count : ℕ)
  (toucan_consumption : ℕ)
  (days_in_week : ℕ)
  (h1 : parakeet_count = 3)
  (h2 : parakeet_consumption = 2)
  (h3 : parrot_count = 2)
  (h4 : parrot_consumption = 14)
  (h5 : finch_count = 4)
  (h6 : canary_count = 5)
  (h7 : canary_consumption = 3)
  (h8 : african_grey_count = 2)
  (h9 : african_grey_consumption = 18)
  (h10 : toucan_count = 3)
  (h11 : toucan_consumption = 25)
  (h12 : days_in_week = 7) :
  (parakeet_count * parakeet_consumption +
   parrot_count * parrot_consumption +
   finch_count * (parakeet_consumption / 2) +
   canary_count * canary_consumption +
   african_grey_count * african_grey_consumption +
   toucan_count * toucan_consumption) * days_in_week = 1148 := by
  sorry

end NUMINAMATH_CALUDE_peters_birdseed_for_week_l913_91375


namespace NUMINAMATH_CALUDE_not_equal_to_three_halves_l913_91368

theorem not_equal_to_three_halves : ∃ x : ℚ, x ≠ (3/2 : ℚ) ∧
  (x = (5/3 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) ∧
  ((3/2 : ℚ) = (3/2 : ℚ)) ∧
  ((7/4 : ℚ) = (3/2 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_not_equal_to_three_halves_l913_91368


namespace NUMINAMATH_CALUDE_common_divisors_45_48_l913_91393

theorem common_divisors_45_48 : Finset.card (Finset.filter (fun d => d ∣ 48) (Nat.divisors 45)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_45_48_l913_91393


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l913_91313

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

/-- The theorem stating that the given equation represents the circle with the specified properties -/
theorem circle_tangent_to_line :
  ∀ x y : ℝ,
  circle_equation x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 ∧
    r = |3 * circle_center.1 - 4 * circle_center.2 + 5| / 5) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l913_91313


namespace NUMINAMATH_CALUDE_four_intersections_implies_a_range_l913_91398

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs x + a - 1

-- State the theorem
theorem four_intersections_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  1 < a ∧ a < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_four_intersections_implies_a_range_l913_91398


namespace NUMINAMATH_CALUDE_commission_rate_is_four_percent_l913_91374

/-- Represents the commission rate as a real number between 0 and 1 -/
def CommissionRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The fixed salary option -/
def fixed_salary : ℝ := 1800

/-- The base salary in the commission option -/
def base_salary : ℝ := 1600

/-- The sales amount at which both options are equal -/
def equal_sales : ℝ := 5000

/-- Calculate the total earnings for the commission option -/
def commission_earnings (rate : CommissionRate) (sales : ℝ) : ℝ :=
  base_salary + (rate.val * sales)

/-- Theorem stating that the commission rate is 4% -/
theorem commission_rate_is_four_percent :
  ∃ (rate : CommissionRate), 
    (commission_earnings rate equal_sales = fixed_salary) ∧ 
    (rate.val = 0.04) := by
  sorry

end NUMINAMATH_CALUDE_commission_rate_is_four_percent_l913_91374


namespace NUMINAMATH_CALUDE_sqrt_pattern_l913_91337

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = ((n + 1) * Real.sqrt (n + 2)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l913_91337


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l913_91378

/-- Triangle with side lengths --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle --/
structure InscribedRectangle where
  base : ℝ

/-- The area of the inscribed rectangle as a function of its base --/
def rectangleArea (t : Triangle) (r : InscribedRectangle) : ℝ → ℝ :=
  fun ω => α * ω - β * ω^2
  where
    α : ℝ := sorry
    β : ℝ := sorry

theorem inscribed_rectangle_area_coefficient (t : Triangle) (r : InscribedRectangle) :
  t.a = 15 ∧ t.b = 34 ∧ t.c = 21 →
  ∃ α β : ℝ, (∀ ω : ℝ, rectangleArea t r ω = α * ω - β * ω^2) ∧ β = 5/41 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l913_91378


namespace NUMINAMATH_CALUDE_proportional_relationship_l913_91390

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 7 when z = 16, prove that x = 7/9 when z = 144 -/
theorem proportional_relationship (k m : ℝ) (h1 : k > 0) (h2 : m > 0) : 
  (∀ x y z : ℝ, x = k * y^2 ∧ y = m / Real.sqrt z → 
    (x = 7 ∧ z = 16 → x * z = 112) ∧
    (z = 144 → x = 7/9)) := by
  sorry

end NUMINAMATH_CALUDE_proportional_relationship_l913_91390


namespace NUMINAMATH_CALUDE_sum_first_three_eq_18_l913_91359

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  fifth_term_eq_15 : a + 4 * d = 15
  d_eq_3 : d = 3

/-- The sum of the first three terms of the arithmetic sequence -/
def sum_first_three (seq : ArithmeticSequence) : ℕ :=
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d)

/-- Theorem stating that the sum of the first three terms is 18 -/
theorem sum_first_three_eq_18 (seq : ArithmeticSequence) :
  sum_first_three seq = 18 := by
  sorry

#eval sum_first_three ⟨3, 3, rfl, rfl⟩

end NUMINAMATH_CALUDE_sum_first_three_eq_18_l913_91359


namespace NUMINAMATH_CALUDE_expression_equality_l913_91341

theorem expression_equality : 
  (|(-1)|^2023 : ℝ) + (Real.sqrt 3)^2 - 2 * Real.sin (π / 6) + (1 / 2)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l913_91341


namespace NUMINAMATH_CALUDE_crayons_distribution_l913_91314

theorem crayons_distribution (total_crayons : ℝ) (x : ℝ) : 
  total_crayons = 210 →
  x / total_crayons = 1 / 30 →
  30 * x = 0.7 * total_crayons →
  x = 4.9 := by
sorry

end NUMINAMATH_CALUDE_crayons_distribution_l913_91314


namespace NUMINAMATH_CALUDE_runner_time_difference_l913_91322

theorem runner_time_difference 
  (x y : ℝ) 
  (h1 : y - x / 2 = 12) 
  (h2 : x - y / 2 = 36) : 
  2 * y - 2 * x = -16 := by
sorry

end NUMINAMATH_CALUDE_runner_time_difference_l913_91322


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l913_91330

theorem sine_cosine_sum (α : Real) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l913_91330


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9871_l913_91354

theorem largest_prime_factor_of_9871 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9871 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9871 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9871_l913_91354


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_l913_91328

theorem recurring_decimal_fraction :
  (5 : ℚ) / 33 / ((2401 : ℚ) / 999) = 4995 / 79233 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_l913_91328


namespace NUMINAMATH_CALUDE_function_nonnegative_iff_a_in_range_l913_91312

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 2

-- Define the theorem
theorem function_nonnegative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a x ≥ 0) ↔ a ∈ [1, 5] := by sorry

end NUMINAMATH_CALUDE_function_nonnegative_iff_a_in_range_l913_91312


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l913_91369

theorem ratio_of_numbers (a b : ℕ) (h1 : a = 45) (h2 : b = 60) (h3 : Nat.lcm a b = 180) :
  (a : ℚ) / b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l913_91369


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l913_91325

/-- The set of integers between 1 and 2^30 with exactly two 1s in their binary expansions -/
def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2^30 ∧ (∃ j k : ℕ, j < k ∧ k < 30 ∧ n = 2^j + 2^k)}

/-- The number of elements in T -/
def T_count : ℕ := 435

/-- The number of elements in T divisible by 15 -/
def T_div15_count : ℕ := 28

theorem probability_divisible_by_15 :
  (T_div15_count : ℚ) / T_count = 28 / 435 ∧ 28 + 435 = 463 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l913_91325


namespace NUMINAMATH_CALUDE_y_squared_value_l913_91382

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l913_91382


namespace NUMINAMATH_CALUDE_combined_weight_is_6600_l913_91388

/-- The weight of the elephant in tons -/
def elephant_weight_tons : ℝ := 3

/-- The weight of a ton in pounds -/
def pounds_per_ton : ℝ := 2000

/-- The percentage of the elephant's weight that the donkey weighs less -/
def donkey_weight_percentage : ℝ := 0.9

/-- The combined weight of the elephant and donkey in pounds -/
def combined_weight_pounds : ℝ :=
  elephant_weight_tons * pounds_per_ton +
  elephant_weight_tons * pounds_per_ton * (1 - donkey_weight_percentage)

theorem combined_weight_is_6600 :
  combined_weight_pounds = 6600 :=
sorry

end NUMINAMATH_CALUDE_combined_weight_is_6600_l913_91388


namespace NUMINAMATH_CALUDE_arrangement_theorem_l913_91332

def num_girls : ℕ := 3
def num_boys : ℕ := 5

def arrangements_girls_together : ℕ := 4320
def arrangements_girls_separate : ℕ := 14400

theorem arrangement_theorem :
  (num_girls = 3 ∧ num_boys = 5) →
  (arrangements_girls_together = 4320 ∧ arrangements_girls_separate = 14400) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l913_91332


namespace NUMINAMATH_CALUDE_quadratic_coefficients_theorem_l913_91363

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of possible coefficient triples (a, b, c) for the quadratic function -/
def PossibleCoefficients : Set (ℝ × ℝ × ℝ) :=
  {(4, -16, 14), (2, -6, 2), (2, -10, 10)}

theorem quadratic_coefficients_theorem (a b c : ℝ) :
  a > 0 ∧
  (∀ x ∈ ({1, 2, 3} : Set ℝ), |QuadraticFunction a b c x| = 2) →
  (a, b, c) ∈ PossibleCoefficients := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_theorem_l913_91363


namespace NUMINAMATH_CALUDE_m_geq_n_l913_91339

theorem m_geq_n (a x : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ 4 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_m_geq_n_l913_91339


namespace NUMINAMATH_CALUDE_solve_linear_equation_l913_91343

theorem solve_linear_equation (x : ℝ) :
  3 + 5 * x = 28 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l913_91343


namespace NUMINAMATH_CALUDE_emily_holidays_l913_91307

/-- The number of holidays Emily takes in a year -/
def holidays_per_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem: Emily takes 24 holidays in a year -/
theorem emily_holidays :
  holidays_per_year 2 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_holidays_l913_91307


namespace NUMINAMATH_CALUDE_product_evaluation_l913_91301

theorem product_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + y^2 + z^2)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x^2*y^2 + y^2*z^2 + z^2*x^2)⁻¹ * ((x*y)⁻¹ + (y*z)⁻¹ + (z*x)⁻¹)) =
  ((x*y + y*z + z*x) * (x + y + z)) / (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) :=
by sorry

end NUMINAMATH_CALUDE_product_evaluation_l913_91301


namespace NUMINAMATH_CALUDE_cube_root_of_456533_l913_91379

theorem cube_root_of_456533 (z : ℤ) :
  z^3 = 456533 → z = 77 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_456533_l913_91379


namespace NUMINAMATH_CALUDE_two_points_same_color_distance_l913_91367

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) :
  ∃ (c : Color) (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end NUMINAMATH_CALUDE_two_points_same_color_distance_l913_91367


namespace NUMINAMATH_CALUDE_triangle_perimeter_sum_l913_91361

theorem triangle_perimeter_sum : 
  ∀ (a b c d e : ℝ),
  a = 6 ∧ b = 8 ∧ 
  c^2 + d^2 = e^2 ∧
  (1/2) * c * d = (1/2) * (1/2) * a * b →
  a + b + (a^2 + b^2).sqrt + c + d + e = 24 + 6 * Real.sqrt 3 + 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_sum_l913_91361


namespace NUMINAMATH_CALUDE_optimal_cup_purchase_l913_91335

/-- Represents the profit optimization problem for cup sales --/
structure CupSalesProblem where
  costA : ℕ
  priceA : ℕ
  costB : ℕ
  priceB : ℕ
  totalCups : ℕ
  budget : ℕ

/-- Calculates the profit for a given number of cup A --/
def profit (p : CupSalesProblem) (x : ℕ) : ℤ :=
  (p.priceA - p.costA) * x + (p.priceB - p.costB) * (p.totalCups - x)

/-- Checks if the purchase is within budget --/
def withinBudget (p : CupSalesProblem) (x : ℕ) : Prop :=
  p.costA * x + p.costB * (p.totalCups - x) ≤ p.budget

/-- Theorem stating the optimal solution and maximum profit --/
theorem optimal_cup_purchase (p : CupSalesProblem) 
  (h1 : p.costA = 100)
  (h2 : p.priceA = 150)
  (h3 : p.costB = 85)
  (h4 : p.priceB = 120)
  (h5 : p.totalCups = 160)
  (h6 : p.budget = 15000) :
  ∃ (x : ℕ), x = 93 ∧ 
             withinBudget p x ∧ 
             profit p x = 6995 ∧ 
             ∀ (y : ℕ), withinBudget p y → profit p y ≤ profit p x :=
by sorry

end NUMINAMATH_CALUDE_optimal_cup_purchase_l913_91335


namespace NUMINAMATH_CALUDE_min_value_expression_l913_91349

theorem min_value_expression (x : ℝ) (hx : x > 0) : 3 * x + 2 / x^5 + 3 / x ≥ 8 ∧
  (3 * x + 2 / x^5 + 3 / x = 8 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l913_91349
