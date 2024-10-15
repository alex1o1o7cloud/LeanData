import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l32_3234

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     b = -12 ∧ 
     c = 24) ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (a * (2 + 2 * Complex.I)^2 + b * (2 + 2 * Complex.I) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l32_3234


namespace NUMINAMATH_CALUDE_average_equation_solution_l32_3239

theorem average_equation_solution (x : ℝ) : 
  (1/4 : ℝ) * ((x + 8) + (7*x - 3) + (3*x + 10) + (-x + 6)) = 5*x - 4 → x = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l32_3239


namespace NUMINAMATH_CALUDE_triangle_side_count_is_35_l32_3254

/-- The number of integer values for the third side of a triangle with two sides of length 18 and 45 -/
def triangle_side_count : ℕ :=
  let possible_x := Finset.filter (fun x : ℕ =>
    x > 27 ∧ x < 63 ∧ x + 18 > 45 ∧ x + 45 > 18 ∧ 18 + 45 > x) (Finset.range 100)
  possible_x.card

theorem triangle_side_count_is_35 : triangle_side_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_count_is_35_l32_3254


namespace NUMINAMATH_CALUDE_find_y_value_l32_3233

theorem find_y_value : ∃ y : ℝ, (15^2 * y^3) / 256 = 450 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l32_3233


namespace NUMINAMATH_CALUDE_locus_is_circle_l32_3214

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse (F₁ F₂ : ℝ × ℝ) where
  a : ℝ
  h : a > 0

/-- A point P on the ellipse -/
def PointOnEllipse (e : Ellipse F₁ F₂) (P : ℝ × ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * e.a

/-- The point Q extended from F₁P such that |PQ| = |PF₂| -/
def ExtendedPoint (P F₁ F₂ : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

/-- The theorem stating that the locus of Q is a circle -/
theorem locus_is_circle (F₁ F₂ : ℝ × ℝ) (e : Ellipse F₁ F₂) :
  ∀ P Q : ℝ × ℝ, PointOnEllipse e P → ExtendedPoint P F₁ F₂ Q →
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, dist Q center = radius :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l32_3214


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l32_3203

/-- Represents the number of male athletes in a stratified sample -/
def male_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (male_athletes * sample_size) / total_athletes

/-- Theorem: In a stratified sampling of 28 athletes from a team of 98 athletes (56 male and 42 female),
    the number of male athletes in the sample should be 16. -/
theorem stratified_sampling_male_athletes :
  male_athletes_in_sample 98 56 28 = 16 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l32_3203


namespace NUMINAMATH_CALUDE_lilia_earnings_l32_3289

/-- Represents Lilia's peach selling scenario -/
structure PeachSale where
  total : Nat
  sold_to_friends : Nat
  price_friends : Real
  sold_to_relatives : Nat
  price_relatives : Real
  kept : Nat

/-- Calculates the total earnings from selling peaches -/
def total_earnings (sale : PeachSale) : Real :=
  sale.sold_to_friends * sale.price_friends + sale.sold_to_relatives * sale.price_relatives

/-- Theorem stating that Lilia's earnings from selling 14 peaches is $25 -/
theorem lilia_earnings (sale : PeachSale) 
  (h1 : sale.total = 15)
  (h2 : sale.sold_to_friends = 10)
  (h3 : sale.price_friends = 2)
  (h4 : sale.sold_to_relatives = 4)
  (h5 : sale.price_relatives = 1.25)
  (h6 : sale.kept = 1)
  (h7 : sale.sold_to_friends + sale.sold_to_relatives + sale.kept = sale.total) :
  total_earnings sale = 25 := by
  sorry

end NUMINAMATH_CALUDE_lilia_earnings_l32_3289


namespace NUMINAMATH_CALUDE_project_assignment_count_l32_3252

/-- The number of ways to assign projects to teams --/
def assign_projects (total_projects : ℕ) (num_teams : ℕ) (max_for_one_team : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given conditions --/
theorem project_assignment_count :
  assign_projects 5 3 2 = 130 :=
sorry

end NUMINAMATH_CALUDE_project_assignment_count_l32_3252


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l32_3219

/-- Represents the inventory composition and new release percentages for a bookstore. -/
structure BookstoreInventory where
  historicalFictionPercentage : Float
  scienceFictionPercentage : Float
  biographiesPercentage : Float
  mysteryNovelsPercentage : Float
  historicalFictionNewReleasePercentage : Float
  scienceFictionNewReleasePercentage : Float
  biographiesNewReleasePercentage : Float
  mysteryNovelsNewReleasePercentage : Float

/-- Calculates the fraction of all new releases that are historical fiction new releases. -/
def historicalFictionNewReleasesFraction (inventory : BookstoreInventory) : Float :=
  let totalNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage +
    inventory.scienceFictionPercentage * inventory.scienceFictionNewReleasePercentage +
    inventory.biographiesPercentage * inventory.biographiesNewReleasePercentage +
    inventory.mysteryNovelsPercentage * inventory.mysteryNovelsNewReleasePercentage
  let historicalFictionNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage
  historicalFictionNewReleases / totalNewReleases

/-- Theorem stating that the fraction of all new releases that are historical fiction new releases is 9/20. -/
theorem historical_fiction_new_releases_fraction :
  let inventory := BookstoreInventory.mk 0.40 0.25 0.15 0.20 0.45 0.30 0.50 0.35
  historicalFictionNewReleasesFraction inventory = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l32_3219


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l32_3232

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3)
  (h2 : a 11 = 15)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 1 = 0 ∧ a 2 - a 1 = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l32_3232


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l32_3280

/-- The number of values of b for which the line y = 2x + b passes through
    the vertex of the parabola y = x^2 - 2bx + b^2 is exactly 1. -/
theorem line_passes_through_parabola_vertex :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 - 2 * b * x + b^2) →
    (x = b ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l32_3280


namespace NUMINAMATH_CALUDE_expand_product_l32_3269

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l32_3269


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l32_3277

/-- Given a square with side length 1, divided into one triangle and three trapezoids
    by joining the center to points on each side, where these points divide each side
    into segments of length 1/4 and 3/4, and each section has equal area,
    prove that the length of the longer parallel side of the trapezoids is 3/4. -/
theorem trapezoid_longer_side_length (square_side : ℝ) (segment_short : ℝ) (segment_long : ℝ)
  (h_square_side : square_side = 1)
  (h_segment_short : segment_short = 1/4)
  (h_segment_long : segment_long = 3/4)
  (h_segments_sum : segment_short + segment_long = square_side)
  (h_equal_areas : ∀ section_area : ℝ, section_area = (square_side^2) / 4) :
  ∃ x : ℝ, x = 3/4 ∧ x = segment_long :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l32_3277


namespace NUMINAMATH_CALUDE_school_boys_count_l32_3288

/-- The percentage of boys who are Muslims -/
def muslim_percentage : ℚ := 44 / 100

/-- The percentage of boys who are Hindus -/
def hindu_percentage : ℚ := 28 / 100

/-- The percentage of boys who are Sikhs -/
def sikh_percentage : ℚ := 10 / 100

/-- The number of boys belonging to other communities -/
def other_communities : ℕ := 153

/-- The total number of boys in the school -/
def total_boys : ℕ := 850

theorem school_boys_count :
  (1 - (muslim_percentage + hindu_percentage + sikh_percentage)) * (total_boys : ℚ) = other_communities := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l32_3288


namespace NUMINAMATH_CALUDE_fruit_punch_water_amount_l32_3257

/-- Represents the ratio of ingredients in the fruit punch -/
structure PunchRatio :=
  (water : ℚ)
  (orange_juice : ℚ)
  (cranberry_juice : ℚ)

/-- Calculates the amount of water needed for a given amount of punch -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.orange_juice + ratio.cranberry_juice
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Theorem stating the amount of water needed for the fruit punch -/
theorem fruit_punch_water_amount :
  let ratio : PunchRatio := ⟨5, 2, 1⟩
  let total_gallons : ℚ := 3
  let quarts_per_gallon : ℚ := 4
  water_needed ratio total_gallons quarts_per_gallon = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_water_amount_l32_3257


namespace NUMINAMATH_CALUDE_negation_of_proposition_l32_3258

theorem negation_of_proposition :
  (¬ ∀ a : ℕ+, 2^(a : ℕ) ≥ (a : ℕ)^2) ↔ (∃ a : ℕ+, 2^(a : ℕ) < (a : ℕ)^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l32_3258


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l32_3205

theorem rectangular_prism_volume 
  (m n Q : ℝ) 
  (m_pos : m > 0) 
  (n_pos : n > 0) 
  (Q_pos : Q > 0) : 
  let base_ratio := m / n
  let diagonal_area := Q
  let volume := (m * n * Q * Real.sqrt Q) / (m^2 + n^2)
  ∃ (a b h : ℝ), 
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    a / b = base_ratio ∧
    a * a + b * b = Q ∧
    h * h = Q ∧
    a * b * h = volume :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l32_3205


namespace NUMINAMATH_CALUDE_max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l32_3290

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := C p.1 p.2

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop := p.1 = -q.1 ∧ p.2 = -q.2

-- Statement 1: Maximum distance between symmetric points
theorem max_distance_symmetric_points :
  ∀ A B : ℝ × ℝ, on_ellipse A → on_ellipse B → symmetric_wrt_origin A B →
  ‖A - B‖ ≤ 10 :=
sorry

-- Statement 2: Constant sum of distances to foci
theorem constant_sum_distances_to_foci :
  ∀ A : ℝ × ℝ, on_ellipse A →
  ‖A - F₁‖ + ‖A - F₂‖ = 10 :=
sorry

-- Statement 3: Ratio of focal length to minor axis
theorem focal_length_to_minor_axis_ratio :
  ‖F₁ - F₂‖ / 8 = 3/4 :=
sorry

-- Statement 4: No point with perpendicular lines to foci
theorem no_perpendicular_lines_to_foci :
  ¬ ∃ A : ℝ × ℝ, on_ellipse A ∧ 
  (A - F₁) • (A - F₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l32_3290


namespace NUMINAMATH_CALUDE_room_tiles_count_l32_3278

/-- Calculates the least number of square tiles required to cover a rectangular floor. -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with given dimensions, 153 square tiles are required. -/
theorem room_tiles_count :
  leastSquareTiles 816 432 = 153 := by
  sorry

#eval leastSquareTiles 816 432

end NUMINAMATH_CALUDE_room_tiles_count_l32_3278


namespace NUMINAMATH_CALUDE_probability_odd_and_multiple_of_3_l32_3210

/-- Represents a fair die with n sides -/
structure Die (n : ℕ) where
  sides : Finset (Fin n)
  fair : sides.card = n

/-- The event of rolling an odd number on a die -/
def oddEvent (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 2 = 1)

/-- The event of rolling a multiple of 3 on a die -/
def multipleOf3Event (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 3 = 0)

/-- The probability of an event occurring on a fair die -/
def probability (d : Die n) (event : Finset (Fin n)) : ℚ :=
  event.card / d.sides.card

theorem probability_odd_and_multiple_of_3 
  (d8 : Die 8) 
  (d12 : Die 12) : 
  probability d8 (oddEvent d8) * probability d12 (multipleOf3Event d12) = 1/6 := by
sorry

end NUMINAMATH_CALUDE_probability_odd_and_multiple_of_3_l32_3210


namespace NUMINAMATH_CALUDE_xiaogang_dart_game_l32_3206

theorem xiaogang_dart_game :
  ∀ (x y z : ℕ),
    x + y + z > 11 →
    8 * x + 9 * y + 10 * z = 100 →
    (x + y + z = 12 ∧ (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_xiaogang_dart_game_l32_3206


namespace NUMINAMATH_CALUDE_equation_always_has_two_solutions_l32_3200

theorem equation_always_has_two_solutions (b : ℝ) (h : 1 ≤ b ∧ b ≤ 25) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^4 + 36*b^2 = (9*b^2 - 15*b)*x₁^2 ∧
  x₂^4 + 36*b^2 = (9*b^2 - 15*b)*x₂^2 :=
sorry

end NUMINAMATH_CALUDE_equation_always_has_two_solutions_l32_3200


namespace NUMINAMATH_CALUDE_modulus_of_z_l32_3262

-- Define the complex number z
def z : ℂ := sorry

-- State the given equation
axiom z_equation : z^2 + z = 1 - 3*Complex.I

-- Define the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l32_3262


namespace NUMINAMATH_CALUDE_binomial_12_11_l32_3263

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l32_3263


namespace NUMINAMATH_CALUDE_compound_proposition_negation_l32_3293

theorem compound_proposition_negation (p q : Prop) : 
  ¬((p ∧ q → false) → (¬p → false) ∧ (¬q → false)) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_negation_l32_3293


namespace NUMINAMATH_CALUDE_only_paintable_number_l32_3230

/-- Represents a painting configuration for the railings. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a given railing number is painted by Harold. -/
def paintedByHarold (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1 + k * config.h

/-- Checks if a given railing number is painted by Tanya. -/
def paintedByTanya (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 + k * config.t

/-- Checks if a given railing number is painted by Ulysses. -/
def paintedByUlysses (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 + k * config.u

/-- Checks if every railing is painted exactly once. -/
def validPainting (config : PaintConfig) : Prop :=
  ∀ n : ℕ+, (paintedByHarold config n ∨ paintedByTanya config n ∨ paintedByUlysses config n) ∧
            ¬(paintedByHarold config n ∧ paintedByTanya config n) ∧
            ¬(paintedByHarold config n ∧ paintedByUlysses config n) ∧
            ¬(paintedByTanya config n ∧ paintedByUlysses config n)

/-- Computes the paintable number for a given configuration. -/
def paintableNumber (config : PaintConfig) : ℕ :=
  100 * config.h + 10 * config.t + config.u

/-- Theorem stating that 453 is the only paintable number. -/
theorem only_paintable_number :
  ∃! n : ℕ, n = 453 ∧ ∃ config : PaintConfig, validPainting config ∧ paintableNumber config = n :=
sorry

end NUMINAMATH_CALUDE_only_paintable_number_l32_3230


namespace NUMINAMATH_CALUDE_crackers_box_sleeves_l32_3267

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights the crackers last -/
def num_nights : ℕ := 56

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

theorem crackers_box_sleeves :
  sleeves_per_box = 4 :=
sorry

end NUMINAMATH_CALUDE_crackers_box_sleeves_l32_3267


namespace NUMINAMATH_CALUDE_farmer_seeds_sowed_l32_3270

/-- The number of buckets of seeds sowed by a farmer -/
def seeds_sowed (initial final : ℝ) : ℝ := initial - final

/-- Theorem stating that the farmer sowed 2.75 buckets of seeds -/
theorem farmer_seeds_sowed :
  seeds_sowed 8.75 6 = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_farmer_seeds_sowed_l32_3270


namespace NUMINAMATH_CALUDE_line_parameterization_l32_3253

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 40 ↔ ∃ t, x = g t ∧ y = 20 * t - 14) →
  ∀ t, g t = 10 * t + 13 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l32_3253


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l32_3237

/-- The x-intercept of the line 2x - 4y = 12 is 6 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x - 4 * y = 12 → y = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l32_3237


namespace NUMINAMATH_CALUDE_simon_beach_treasures_l32_3211

/-- Represents the number of treasures Simon collected on the beach. -/
def beach_treasures (sand_dollars : ℕ) (glass_multiplier : ℕ) (shell_multiplier : ℕ) : ℕ :=
  let glass := sand_dollars * glass_multiplier
  let shells := glass * shell_multiplier
  sand_dollars + glass + shells

/-- Proves that Simon collected 190 treasures on the beach. -/
theorem simon_beach_treasures :
  beach_treasures 10 3 5 = 190 := by
  sorry

end NUMINAMATH_CALUDE_simon_beach_treasures_l32_3211


namespace NUMINAMATH_CALUDE_investment_problem_l32_3291

theorem investment_problem (x y : ℝ) (h1 : x + y = 3000) 
  (h2 : 0.08 * x + 0.05 * y = 490 ∨ 0.08 * y + 0.05 * x = 490) : 
  x + y = 8000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l32_3291


namespace NUMINAMATH_CALUDE_bucket_capacity_l32_3297

/-- The capacity of a bucket in litres, given that when it is 2/3 full, it contains 9 litres of maple syrup. -/
theorem bucket_capacity : ℝ := by
  -- Define the capacity of the bucket
  let C : ℝ := 13.5

  -- Define the fraction of the bucket that is full
  let fraction_full : ℝ := 2/3

  -- Define the current volume of maple syrup
  let current_volume : ℝ := 9

  -- State that the current volume is equal to the fraction of the capacity
  have h1 : fraction_full * C = current_volume := by sorry

  -- Prove that the capacity is indeed 13.5 litres
  have h2 : C = 13.5 := by sorry

  -- Return the capacity
  exact C

end NUMINAMATH_CALUDE_bucket_capacity_l32_3297


namespace NUMINAMATH_CALUDE_equation_solution_l32_3248

theorem equation_solution : ∃ x : ℝ, 24 - 6 = 3 + x ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l32_3248


namespace NUMINAMATH_CALUDE_fidos_yard_area_l32_3242

theorem fidos_yard_area (s : ℝ) (h : s > 0) :
  let r := s / 2
  let area_circle := π * r^2
  let area_square := s^2
  let fraction := area_circle / area_square
  ∃ (a b : ℝ), fraction = (Real.sqrt a / b) * π ∧ a * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_fidos_yard_area_l32_3242


namespace NUMINAMATH_CALUDE_inequalities_theorem_l32_3296

theorem inequalities_theorem :
  (∀ (a b c d : ℝ), a > b → c > d → a - d > b - c) ∧
  (∀ (a b : ℝ), 1/a < 1/b → 1/b < 0 → a*b < b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l32_3296


namespace NUMINAMATH_CALUDE_line_equation_intersection_condition_max_value_condition_l32_3240

-- Define the parabola and line
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1
def line (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Theorem 1: Line equation
theorem line_equation :
  ∃ k b : ℝ, (line k b (-2) = -5/2) ∧ (line k b 3 = 0) →
  (k = 1/2 ∧ b = -3/2) :=
sorry

-- Theorem 2: Intersection condition
theorem intersection_condition (a : ℝ) :
  a ≠ 0 →
  (∃ x : ℝ, parabola a x = line (1/2) (-3/2) x) ↔
  (a ≤ 9/8) :=
sorry

-- Theorem 3: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x : ℝ, m ≤ x ∧ x ≤ m + 2 →
    parabola (-1) x ≤ -4) ∧
    (∃ x : ℝ, m ≤ x ∧ x ≤ m + 2 ∧ parabola (-1) x = -4) →
  (m = -3 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_intersection_condition_max_value_condition_l32_3240


namespace NUMINAMATH_CALUDE_apple_distribution_l32_3282

/-- The number of ways to distribute n items among k people with a minimum of m items each -/
def distribution_ways (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 253 ways to distribute 30 apples among 3 people with at least 3 apples each -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l32_3282


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l32_3243

def a : Fin 3 → ℝ := ![1, 5, 2]
def b : Fin 3 → ℝ := ![-1, 1, -1]
def c : Fin 3 → ℝ := ![1, 1, 1]

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l32_3243


namespace NUMINAMATH_CALUDE_xyz_value_l32_3256

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l32_3256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_value_b_l32_3212

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def cosine_condition (t : Triangle) : Prop :=
  2 * Real.cos t.B * (t.c * Real.cos t.A + t.a * Real.cos t.C) = t.b

def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2

-- Theorem 1: Arithmetic sequence
theorem arithmetic_sequence (t : Triangle) 
  (h1 : triangle_condition t) (h2 : cosine_condition t) : 
  ∃ r : Real, t.A = t.B - r ∧ t.C = t.B + r :=
sorry

-- Theorem 2: Minimum value of b
theorem min_value_b (t : Triangle) 
  (h1 : triangle_condition t) (h2 : area_condition t) : 
  t.b ≥ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_value_b_l32_3212


namespace NUMINAMATH_CALUDE_josephs_total_cards_l32_3249

/-- The number of cards in a standard deck -/
def cards_per_deck : ℕ := 52

/-- The number of decks Joseph has -/
def josephs_decks : ℕ := 4

/-- Theorem: Joseph has 208 cards in total -/
theorem josephs_total_cards : 
  josephs_decks * cards_per_deck = 208 := by
  sorry

end NUMINAMATH_CALUDE_josephs_total_cards_l32_3249


namespace NUMINAMATH_CALUDE_fraction_division_evaluate_fraction_l32_3286

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c :=
by sorry

theorem evaluate_fraction :
  (4 : ℚ) / ((8 : ℚ) / 13) = 13 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_evaluate_fraction_l32_3286


namespace NUMINAMATH_CALUDE_computer_profit_profit_function_max_profit_l32_3227

/-- Profit from selling computers -/
theorem computer_profit (profit_A profit_B : ℚ) : 
  (10 * profit_A + 20 * profit_B = 4000) →
  (20 * profit_A + 10 * profit_B = 3500) →
  (profit_A = 100 ∧ profit_B = 150) :=
sorry

/-- Functional relationship between total profit and number of type A computers -/
theorem profit_function (x y : ℚ) :
  (x ≥ 0 ∧ x ≤ 100) →
  (y = 100 * x + 150 * (100 - x)) →
  (y = -50 * x + 15000) :=
sorry

/-- Maximum profit when purchasing at least 20 units of type A -/
theorem max_profit (x y : ℚ) :
  (x ≥ 20 ∧ x ≤ 100) →
  (y = -50 * x + 15000) →
  (∀ z, z ≥ 20 ∧ z ≤ 100 → -50 * z + 15000 ≤ 14000) :=
sorry

end NUMINAMATH_CALUDE_computer_profit_profit_function_max_profit_l32_3227


namespace NUMINAMATH_CALUDE_smallest_period_scaled_l32_3231

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 15) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f ((x - a) / 3) = f (x / 3)) ∧
    ∀ b, b > 0 → (∀ x, f ((x - b) / 3) = f (x / 3)) → a ≤ b :=
  sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_l32_3231


namespace NUMINAMATH_CALUDE_ammeter_readings_sum_l32_3216

/-- The sum of readings of five ammeters in a specific circuit configuration -/
def sum_of_ammeter_readings (I₁ I₂ I₃ I₄ I₅ : ℝ) : ℝ :=
  I₁ + I₂ + I₃ + I₄ + I₅

/-- Theorem stating the sum of ammeter readings in the given circuit -/
theorem ammeter_readings_sum :
  ∀ (I₁ I₂ I₃ I₄ I₅ : ℝ),
    I₁ = 2 →
    I₂ = I₁ →
    I₃ = I₁ + I₂ →
    I₅ = I₃ + I₁ →
    I₄ = (5/3) * I₅ →
    sum_of_ammeter_readings I₁ I₂ I₃ I₄ I₅ = 24 := by
  sorry


end NUMINAMATH_CALUDE_ammeter_readings_sum_l32_3216


namespace NUMINAMATH_CALUDE_problem_solution_l32_3255

theorem problem_solution : 
  let a := (5 / 6) * 180
  let b := 0.70 * 250
  let diff := a - b
  let c := 0.35 * 480
  diff / c = -0.14880952381 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l32_3255


namespace NUMINAMATH_CALUDE_average_value_function_m_range_l32_3266

/-- Definition of an average value function -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Ioo a b, f x₀ = (f b - f a) / (b - a)

/-- The function we're considering -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ x^2 - m*x - 1

/-- The theorem statement -/
theorem average_value_function_m_range :
  ∀ m : ℝ, is_average_value_function (f m) (-1) 1 → 0 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_average_value_function_m_range_l32_3266


namespace NUMINAMATH_CALUDE_fraction_evaluation_l32_3244

theorem fraction_evaluation : (4 - 3/5) / (3 - 2/7) = 119/95 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l32_3244


namespace NUMINAMATH_CALUDE_erased_numbers_theorem_l32_3250

def sumBetween (a b : ℕ) : ℕ := (b - a - 1) * (a + b) / 2

def sumOutside (a b : ℕ) : ℕ := (2018 * 2019) / 2 - sumBetween a b - a - b

theorem erased_numbers_theorem (a b : ℕ) (ha : a = 673) (hb : b = 1346) :
  2 * sumBetween a b = sumOutside a b := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_theorem_l32_3250


namespace NUMINAMATH_CALUDE_inequality_solution_set_l32_3201

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) ↔ |k * x - 4| ≤ 2) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l32_3201


namespace NUMINAMATH_CALUDE_distance_between_centers_l32_3294

/-- Given a triangle with sides 5, 12, and 13, the distance between the centers
    of its inscribed and circumscribed circles is √65/2 -/
theorem distance_between_centers (a b c : ℝ) (h_sides : a = 5 ∧ b = 12 ∧ c = 13) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt ((circumradius - inradius) ^ 2 + (area / (a * b * c) * (a + b - c) * (b + c - a) * (c + a - b))) = Real.sqrt 65 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l32_3294


namespace NUMINAMATH_CALUDE_decimal_division_l32_3220

theorem decimal_division (x y : ℚ) (hx : x = 0.45) (hy : y = 0.005) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l32_3220


namespace NUMINAMATH_CALUDE_union_A_complement_B_range_of_a_l32_3284

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) (h : A ∩ C a = A) : 1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_range_of_a_l32_3284


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l32_3273

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l32_3273


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l32_3202

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b = f'(2), 
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_function_derivative (a b : ℝ) : 
  let f := fun x : ℝ => a * x^3 + b * x^2 + 3
  let f' := fun x : ℝ => 3 * a * x^2 + 2 * b * x
  (f' 1 = -5 ∧ b = f' 2) → f' 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l32_3202


namespace NUMINAMATH_CALUDE_typing_service_cost_l32_3283

/-- Typing service cost calculation -/
theorem typing_service_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) :
  total_pages = 100 →
  revised_once = 20 →
  revised_twice = 30 →
  revision_cost = 5 →
  total_cost = 1400 →
  ∃ (first_time_cost : ℚ),
    first_time_cost * total_pages + 
    revision_cost * (revised_once + 2 * revised_twice) = total_cost ∧
    first_time_cost = 10 := by
  sorry


end NUMINAMATH_CALUDE_typing_service_cost_l32_3283


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l32_3208

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ≥ 4 * Real.sqrt 3 :=
by sorry

theorem min_value_attainable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l32_3208


namespace NUMINAMATH_CALUDE_proposition_2_proposition_3_l32_3245

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Define the given conditions
variable (m n a b : Line) (α β : Plane)
variable (h_mn_distinct : m ≠ n)
variable (h_αβ_distinct : α ≠ β)
variable (h_a_perp_α : perpendicularLP a α)
variable (h_b_perp_β : perpendicularLP b β)

-- State the theorems to be proved
theorem proposition_2 
  (h_m_parallel_a : parallel m a)
  (h_n_parallel_b : parallel n b)
  (h_α_perp_β : perpendicularPP α β) :
  perpendicular m n :=
sorry

theorem proposition_3
  (h_m_parallel_α : parallelLP m α)
  (h_n_parallel_b : parallel n b)
  (h_α_parallel_β : parallelPP α β) :
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_proposition_2_proposition_3_l32_3245


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l32_3235

theorem sum_of_number_and_its_square : 
  let n : ℕ := 8
  (n + n^2) = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l32_3235


namespace NUMINAMATH_CALUDE_unknown_denomination_is_500_l32_3241

/-- Represents the denomination problem with given conditions --/
structure DenominationProblem where
  total_amount : ℕ
  known_denomination : ℕ
  total_notes : ℕ
  known_denomination_count : ℕ
  (total_amount_check : total_amount = 10350)
  (known_denomination_check : known_denomination = 50)
  (total_notes_check : total_notes = 54)
  (known_denomination_count_check : known_denomination_count = 37)

/-- Theorem stating that the unknown denomination is 500 --/
theorem unknown_denomination_is_500 (p : DenominationProblem) : 
  (p.total_amount - p.known_denomination * p.known_denomination_count) / (p.total_notes - p.known_denomination_count) = 500 :=
sorry

end NUMINAMATH_CALUDE_unknown_denomination_is_500_l32_3241


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l32_3238

/-- The polynomial P(x) = x^2 - x satisfies P(P(x)) = (x^2 - x + 1) P(x) and is the only nonconstant polynomial solution. -/
theorem unique_polynomial_composition (x : ℝ) : ∃! P : ℝ → ℝ, 
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0 ∨ b ≠ 0) ∧
  (∀ x, P (P x) = (x^2 - x + 1) * P x) ∧
  P = fun x ↦ x^2 - x := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l32_3238


namespace NUMINAMATH_CALUDE_investment_value_proof_l32_3292

theorem investment_value_proof (x : ℝ) : 
  (0.07 * x + 0.11 * 1500 = 0.10 * (x + 1500)) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_proof_l32_3292


namespace NUMINAMATH_CALUDE_average_of_numbers_l32_3281

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125831.9 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l32_3281


namespace NUMINAMATH_CALUDE_regular_polyhedra_coloring_l32_3222

structure RegularPolyhedron where
  edges : ℕ
  vertexDegree : ℕ
  faceEdges : ℕ

def isGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  redEdges ≤ p.edges * (p.vertexDegree - 1) / p.vertexDegree

def isCompletelyGoodColoring (p : RegularPolyhedron) (redEdges : ℕ) : Prop :=
  isGoodColoring p redEdges ∧ redEdges < p.edges

def maxGoodColoring (p : RegularPolyhedron) : ℕ :=
  p.edges * (p.vertexDegree - 1) / p.vertexDegree

def maxCompletelyGoodColoring (p : RegularPolyhedron) : ℕ :=
  min (maxGoodColoring p) (p.edges - 1)

def tetrahedron : RegularPolyhedron := ⟨6, 3, 3⟩
def cube : RegularPolyhedron := ⟨12, 3, 4⟩
def octahedron : RegularPolyhedron := ⟨12, 4, 3⟩
def dodecahedron : RegularPolyhedron := ⟨30, 3, 5⟩
def icosahedron : RegularPolyhedron := ⟨30, 5, 3⟩

theorem regular_polyhedra_coloring :
  (maxGoodColoring tetrahedron = maxCompletelyGoodColoring tetrahedron) ∧
  (maxGoodColoring cube = maxCompletelyGoodColoring cube) ∧
  (maxGoodColoring dodecahedron = maxCompletelyGoodColoring dodecahedron) ∧
  (maxGoodColoring octahedron ≠ maxCompletelyGoodColoring octahedron) ∧
  (maxGoodColoring icosahedron ≠ maxCompletelyGoodColoring icosahedron) := by
  sorry

end NUMINAMATH_CALUDE_regular_polyhedra_coloring_l32_3222


namespace NUMINAMATH_CALUDE_triangle_area_l32_3229

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 5, b = 4, and cos(A - B) = 31/32, then the area of the triangle is (15 * √7) / 4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  a = 5 →
  b = 4 →
  Real.cos (A - B) = 31/32 →
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l32_3229


namespace NUMINAMATH_CALUDE_curve_S_properties_l32_3223

-- Define the curve S
def S (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of S
def S' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Define the point P
def P : ℝ × ℝ := (2, -12)

theorem curve_S_properties :
  -- 1. P is the point where the tangent line has the smallest slope
  (∀ x : ℝ, S' P.1 ≤ S' x) ∧
  -- 2. S is symmetric about P
  (∀ x : ℝ, S (P.1 + x) - P.2 = -(S (P.1 - x) - P.2)) :=
sorry

end NUMINAMATH_CALUDE_curve_S_properties_l32_3223


namespace NUMINAMATH_CALUDE_locus_characterization_l32_3261

/-- The locus of points equidistant from A(4, 1) and the y-axis -/
def locus_equation (x y : ℝ) : Prop :=
  (y - 1)^2 = 16 * (x - 2)

/-- A point P(x, y) is equidistant from A(4, 1) and the y-axis -/
def is_equidistant (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = x^2

theorem locus_characterization (x y : ℝ) :
  is_equidistant x y ↔ locus_equation x y := by sorry

end NUMINAMATH_CALUDE_locus_characterization_l32_3261


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l32_3271

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Joan gave to Mike -/
def seashells_given : ℕ := 63

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 16

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_eq_sum : total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l32_3271


namespace NUMINAMATH_CALUDE_counterfeit_coins_l32_3285

def bags : List Nat := [18, 19, 21, 23, 25, 34]

structure Distribution where
  xiaocong : List Nat
  xiaomin : List Nat
  counterfeit : Nat

def isValidDistribution (d : Distribution) : Prop :=
  d.xiaocong.length = 3 ∧
  d.xiaomin.length = 2 ∧
  d.xiaocong.sum = 2 * d.xiaomin.sum ∧
  d.counterfeit ∈ bags ∧
  d.xiaocong.sum + d.xiaomin.sum + d.counterfeit = bags.sum

theorem counterfeit_coins (d : Distribution) :
  isValidDistribution d → d.counterfeit = 23 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_coins_l32_3285


namespace NUMINAMATH_CALUDE_product_even_sum_undetermined_l32_3276

theorem product_even_sum_undetermined (a b : ℤ) : 
  Even (a * b) → (Even (a + b) ∨ Odd (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_product_even_sum_undetermined_l32_3276


namespace NUMINAMATH_CALUDE_range_of_m_l32_3274

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -Real.sqrt (9 - p.1^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the vector from A to P
def AP (m : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - 0, P.2 - m)

-- Define the vector from A to Q
def AQ (m : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - 0, Q.2 - m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∃ P ∈ C, ∃ Q ∈ l, AP m P + AQ m Q = (0, 0)) → m ∈ Set.Icc (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l32_3274


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l32_3268

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l32_3268


namespace NUMINAMATH_CALUDE_locus_of_intersection_points_l32_3275

/-- The locus of intersection points of perpendiculars drawn from a circle's 
    intersections with two perpendicular lines. -/
theorem locus_of_intersection_points (u v x y : ℝ) :
  (u ≠ v ∨ u ≠ -v) →
  (∃ (r : ℝ), r > 0 ∧ r > |u| ∧ r > |v| ∧
    (x - u)^2 / (u^2 - v^2) - (y - v)^2 / (u^2 - v^2) = 1) ∨
  (u = v ∨ u = -v) →
    (x - y) * (x + y) = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_intersection_points_l32_3275


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l32_3204

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (third_group : ℕ)
  (second_group_paintings : ℕ) (third_group_paintings : ℕ) (total_paintings_sold : ℕ)
  (h1 : total_customers = first_group + second_group + third_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 4)
  (h4 : second_group = 12)
  (h5 : third_group = 4)
  (h6 : second_group_paintings = 1)
  (h7 : third_group_paintings = 4)
  (h8 : total_paintings_sold = 36) :
  (total_paintings_sold - (second_group * second_group_paintings + third_group * third_group_paintings)) / first_group = 2 :=
sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l32_3204


namespace NUMINAMATH_CALUDE_value_of_a_l32_3225

theorem value_of_a (U : Set ℝ) (A : Set ℝ) (a : ℝ) : 
  U = {2, 3, a^2 - a - 1} →
  A = {2, 3} →
  U \ A = {1} →
  a = -1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l32_3225


namespace NUMINAMATH_CALUDE_regression_line_equation_l32_3218

/-- Given a regression line with slope 1.23 passing through the point (4, 5),
    prove that its equation is ŷ = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (intercept : ℝ), 
    intercept = center_y - slope * center_x ∧
    intercept = 0.08 ∧
    ∀ (x y : ℝ), y = slope * x + intercept := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l32_3218


namespace NUMINAMATH_CALUDE_product_remainder_ten_l32_3236

theorem product_remainder_ten (a b c : ℕ) (ha : a = 2153) (hb : b = 3491) (hc : c = 925) :
  (a * b * c) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_ten_l32_3236


namespace NUMINAMATH_CALUDE_cody_marbles_l32_3264

def initial_marbles : ℕ := 12
def marbles_given : ℕ := 5

theorem cody_marbles : initial_marbles - marbles_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l32_3264


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l32_3287

theorem sqrt_sum_problem (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l32_3287


namespace NUMINAMATH_CALUDE_smallest_integer_ending_in_9_divisible_by_13_l32_3298

theorem smallest_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by
  use 39
  sorry

end NUMINAMATH_CALUDE_smallest_integer_ending_in_9_divisible_by_13_l32_3298


namespace NUMINAMATH_CALUDE_rectangle_max_area_l32_3295

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l32_3295


namespace NUMINAMATH_CALUDE_probability_theorem_l32_3213

def num_events : ℕ := 5
def prob_success : ℚ := 3/4

theorem probability_theorem :
  (prob_success ^ num_events = 243/1024) ∧
  (1 - (1 - prob_success) ^ num_events = 1023/1024) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l32_3213


namespace NUMINAMATH_CALUDE_otimes_nested_equal_101_l32_3217

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := b^2 + 1

-- Theorem statement
theorem otimes_nested_equal_101 (m : ℚ) : otimes m (otimes m 3) = 101 := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equal_101_l32_3217


namespace NUMINAMATH_CALUDE_store_max_profit_l32_3247

/-- A store selling clothing with the following conditions:
    - Cost price is 60 yuan per item
    - Selling price must not be lower than the cost price
    - Profit must not exceed 40%
    - Sales volume (y) and selling price (x) follow a linear function y = kx + b
    - When x = 80, y = 40
    - When x = 70, y = 50
    - 60 ≤ x ≤ 84
-/
theorem store_max_profit (x : ℝ) (y : ℝ) (k : ℝ) (b : ℝ) (W : ℝ → ℝ) :
  (∀ x, 60 ≤ x ∧ x ≤ 84) →
  (∀ x, y = k * x + b) →
  (80 * k + b = 40) →
  (70 * k + b = 50) →
  (∀ x, W x = (x - 60) * (k * x + b)) →
  (∃ x₀, ∀ x, 60 ≤ x ∧ x ≤ 84 → W x ≤ W x₀) →
  (∃ x₀, W x₀ = 864 ∧ x₀ = 84) := by
  sorry


end NUMINAMATH_CALUDE_store_max_profit_l32_3247


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l32_3265

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l32_3265


namespace NUMINAMATH_CALUDE_jack_apples_l32_3251

/-- Calculates the remaining apples after a series of sales and a gift --/
def remaining_apples (initial : ℕ) (sale1_percent : ℕ) (sale2_percent : ℕ) (sale3_percent : ℕ) (gift : ℕ) : ℕ :=
  let after_sale1 := initial - initial * sale1_percent / 100
  let after_sale2 := after_sale1 - after_sale1 * sale2_percent / 100
  let after_sale3 := after_sale2 - (after_sale2 * sale3_percent / 100)
  after_sale3 - gift

/-- Theorem stating that given the specific conditions, Jack ends up with 75 apples --/
theorem jack_apples : remaining_apples 150 30 20 10 1 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jack_apples_l32_3251


namespace NUMINAMATH_CALUDE_sqrt_4_minus_x_real_range_l32_3259

theorem sqrt_4_minus_x_real_range : 
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 4 - x} = {x : ℝ | x ≤ 4} := by
sorry

end NUMINAMATH_CALUDE_sqrt_4_minus_x_real_range_l32_3259


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l32_3209

-- Problem 1
theorem equation_one_solution (x : ℝ) :
  (2 / x + 1 / (x * (x - 2)) = 5 / (2 * x)) ↔ x = 4 :=
sorry

-- Problem 2
theorem equation_two_no_solution :
  ¬∃ (x : ℝ), (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l32_3209


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l32_3228

theorem nested_fraction_evaluation :
  1 / (3 - 1 / (2 - 1 / (3 - 1 / (2 - 1 / 2)))) = 11 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l32_3228


namespace NUMINAMATH_CALUDE_polyhedron_edge_vertex_relation_l32_3226

/-- Represents a polyhedron with its vertex and edge properties -/
structure Polyhedron where
  /-- p k is the number of vertices where k edges meet -/
  p : ℕ → ℕ
  /-- a is the total number of edges -/
  a : ℕ

/-- The sum of k * p k for all k ≥ 3 equals twice the total number of edges -/
theorem polyhedron_edge_vertex_relation (P : Polyhedron) :
  2 * P.a = ∑' k, k * P.p k := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_vertex_relation_l32_3226


namespace NUMINAMATH_CALUDE_remainder_theorem_l32_3260

theorem remainder_theorem (N : ℤ) (h : N % 100 = 11) : N % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l32_3260


namespace NUMINAMATH_CALUDE_path_count_on_grid_l32_3246

/-- The number of distinct paths on a 6x5 grid from upper left to lower right corner -/
def number_of_paths : ℕ := 126

/-- The number of right moves required to reach the right edge of a 6x5 grid -/
def right_moves : ℕ := 5

/-- The number of down moves required to reach the bottom edge of a 6x5 grid -/
def down_moves : ℕ := 4

/-- The total number of moves (right + down) required to reach the bottom right corner -/
def total_moves : ℕ := right_moves + down_moves

theorem path_count_on_grid : 
  number_of_paths = Nat.choose total_moves right_moves :=
by sorry

end NUMINAMATH_CALUDE_path_count_on_grid_l32_3246


namespace NUMINAMATH_CALUDE_landscape_length_l32_3221

/-- A rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playground_area : ℝ
  length_is_four_times_breadth : length = 4 * breadth
  playground_area_is_1200 : playground_area = 1200
  playground_is_one_third : playground_area = (1/3) * (length * breadth)

/-- The length of the landscape is 120 meters -/
theorem landscape_length (L : Landscape) : L.length = 120 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l32_3221


namespace NUMINAMATH_CALUDE_expression_value_at_one_fifth_l32_3207

theorem expression_value_at_one_fifth :
  let x : ℚ := 1/5
  (x^2 - 4) / (x^2 - 2*x) = 11 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_one_fifth_l32_3207


namespace NUMINAMATH_CALUDE_right_triangle_sets_l32_3279

theorem right_triangle_sets : 
  (¬ (4^2 + 6^2 = 8^2)) ∧ 
  (5^2 + 12^2 = 13^2) ∧ 
  (6^2 + 8^2 = 10^2) ∧ 
  (7^2 + 24^2 = 25^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l32_3279


namespace NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l32_3299

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the condition for three distinct intersection points
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l32_3299


namespace NUMINAMATH_CALUDE_only_1680_is_product_of_four_consecutive_l32_3272

/-- Given a natural number n, returns true if n can be expressed as a product of four consecutive natural numbers. -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) * (k + 2) * (k + 3)

/-- Theorem stating that among 712, 1262, and 1680, only 1680 can be expressed as a product of four consecutive natural numbers. -/
theorem only_1680_is_product_of_four_consecutive :
  ¬ is_product_of_four_consecutive 712 ∧
  ¬ is_product_of_four_consecutive 1262 ∧
  is_product_of_four_consecutive 1680 :=
by sorry

end NUMINAMATH_CALUDE_only_1680_is_product_of_four_consecutive_l32_3272


namespace NUMINAMATH_CALUDE_championship_outcomes_l32_3224

def number_of_competitors : ℕ := 5
def number_of_events : ℕ := 3

theorem championship_outcomes :
  (number_of_competitors ^ number_of_events : ℕ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l32_3224


namespace NUMINAMATH_CALUDE_complex_arithmetic_l32_3215

theorem complex_arithmetic (z : ℂ) (h : z = 1 + I) : (2 / z) + z^2 = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l32_3215
