import Mathlib

namespace cross_in_square_l1486_148682

theorem cross_in_square (s : ℝ) : 
  s > 0 → 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → 
  s = 36 := by
sorry

end cross_in_square_l1486_148682


namespace money_distribution_l1486_148641

/-- Given three people with a total of $4000, where one person has two-thirds of the amount
    the other two have combined, prove that this person has $1600. -/
theorem money_distribution (total : ℚ) (r_share : ℚ) : 
  total = 4000 →
  r_share = (2/3) * (total - r_share) →
  r_share = 1600 := by
  sorry

end money_distribution_l1486_148641


namespace arithmetic_sequence_k_value_l1486_148606

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77)
  (h3 : ∃ k : ℕ, a k = 13) :
  ∃ k : ℕ, a k = 13 ∧ k = 18 := by
  sorry

end arithmetic_sequence_k_value_l1486_148606


namespace flowers_per_bouquet_l1486_148652

theorem flowers_per_bouquet (total_flowers : ℕ) (wilted_flowers : ℕ) (num_bouquets : ℕ) :
  total_flowers = 88 →
  wilted_flowers = 48 →
  num_bouquets = 8 →
  (total_flowers - wilted_flowers) / num_bouquets = 5 :=
by sorry

end flowers_per_bouquet_l1486_148652


namespace kim_nail_polishes_l1486_148668

/-- Given information about nail polishes owned by Kim, Heidi, and Karen, prove that Kim has 12 nail polishes. -/
theorem kim_nail_polishes :
  ∀ (K : ℕ), -- Kim's nail polishes
  (K + 5) + (K - 4) = 25 → -- Heidi and Karen's total
  K = 12 := by
sorry

end kim_nail_polishes_l1486_148668


namespace enjoyable_gameplay_l1486_148688

theorem enjoyable_gameplay (total_hours : ℝ) (boring_percentage : ℝ) (expansion_hours : ℝ) :
  total_hours = 100 ∧ 
  boring_percentage = 80 ∧ 
  expansion_hours = 30 →
  (1 - boring_percentage / 100) * total_hours + expansion_hours = 50 := by
  sorry

end enjoyable_gameplay_l1486_148688


namespace geometric_series_common_ratio_l1486_148628

/-- The common ratio of the infinite geometric series 7/8 - 14/32 + 56/256 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/32
  let a₃ : ℚ := 56/256
  let r : ℚ := a₂ / a₁
  (r = -1/2) ∧ (a₃ / a₂ = r) := by sorry

end geometric_series_common_ratio_l1486_148628


namespace function_properties_l1486_148649

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 4*a*x + b

theorem function_properties :
  ∃ (a b : ℝ),
    -- Condition: f(1) = 3
    f a b 1 = 3 ∧
    -- Condition: f'(1) = 1 (slope of tangent line at x=1)
    f_derivative a b 1 = 1 ∧
    -- Prove: a = 2 and b = 6
    a = 2 ∧ b = 6 ∧
    -- Prove: Range of f(x) on [-1, 4] is [-11, 24]
    (∀ x, -1 ≤ x ∧ x ≤ 4 → -11 ≤ f a b x ∧ f a b x ≤ 24) ∧
    f a b (-1) = -11 ∧ f a b 4 = 24 :=
by sorry

end function_properties_l1486_148649


namespace integer_part_of_sqrt18_minus_2_l1486_148648

theorem integer_part_of_sqrt18_minus_2 :
  ⌊Real.sqrt 18 - 2⌋ = 2 := by
  sorry

end integer_part_of_sqrt18_minus_2_l1486_148648


namespace triangle_angle_measure_l1486_148640

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 6 →
  b = Real.sqrt 3 →
  b + a * (Real.sin C - Real.cos C) = 0 →
  A + B + C = π →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 6 := by
sorry

end triangle_angle_measure_l1486_148640


namespace max_n_is_26_l1486_148614

/-- The number of non-congruent trapezoids formed by four points out of n equally spaced points on a circle's circumference -/
def num_trapezoids (n : ℕ) : ℕ := sorry

/-- The maximum value of n such that the number of non-congruent trapezoids is no more than 2012 -/
def max_n : ℕ := sorry

theorem max_n_is_26 :
  (∀ n : ℕ, n > 0 → num_trapezoids n ≤ 2012) ∧
  (∀ m : ℕ, m > max_n → num_trapezoids m > 2012) ∧
  max_n = 26 := by sorry

end max_n_is_26_l1486_148614


namespace a_upper_bound_l1486_148681

theorem a_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2*x > x^2 + a) → a < -8 := by
  sorry

end a_upper_bound_l1486_148681


namespace cos_240_degrees_l1486_148692

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l1486_148692


namespace scenario_is_simple_random_sampling_l1486_148690

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students -/
structure Population where
  size : ℕ
  is_first_year : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  population : Population
  selection_method : SamplingMethod

/-- The sampling method used in the given scenario -/
def scenario_sampling : Sample where
  size := 20
  population := { size := 200, is_first_year := true }
  selection_method := SamplingMethod.SimpleRandom

/-- Theorem stating that the sampling method used in the scenario is simple random sampling -/
theorem scenario_is_simple_random_sampling :
  scenario_sampling.selection_method = SamplingMethod.SimpleRandom :=
by
  sorry


end scenario_is_simple_random_sampling_l1486_148690


namespace workout_total_weight_l1486_148632

/-- Represents a weightlifting exercise with weight and repetitions -/
structure Exercise where
  weight : ℕ
  reps : ℕ

/-- Calculates the total weight lifted for an exercise -/
def totalWeight (e : Exercise) : ℕ := e.weight * e.reps

/-- Represents a workout session with three exercises -/
structure WorkoutSession where
  chest : Exercise
  back : Exercise
  legs : Exercise

/-- Calculates the grand total weight lifted in a workout session -/
def grandTotalWeight (w : WorkoutSession) : ℕ :=
  totalWeight w.chest + totalWeight w.back + totalWeight w.legs

/-- Theorem: The grand total weight lifted in the given workout session is 2200 pounds -/
theorem workout_total_weight :
  let workout : WorkoutSession := {
    chest := { weight := 90, reps := 8 },
    back := { weight := 70, reps := 10 },
    legs := { weight := 130, reps := 6 }
  }
  grandTotalWeight workout = 2200 := by sorry

end workout_total_weight_l1486_148632


namespace angle_terminal_side_range_l1486_148669

theorem angle_terminal_side_range (θ : Real) (a : Real) :
  (∃ (x y : Real), x = a - 2 ∧ y = a + 2 ∧ x = y * Real.tan θ) →
  Real.cos θ ≤ 0 →
  Real.sin θ > 0 →
  a ∈ Set.Ioo (-2) 2 := by
sorry

end angle_terminal_side_range_l1486_148669


namespace minimum_cost_for_boxes_l1486_148665

/-- The dimensions of a box in inches -/
def box_dimensions : Fin 3 → ℕ
  | 0 => 20
  | 1 => 20
  | 2 => 15
  | _ => 0

/-- The volume of a single box in cubic inches -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The cost of a single box in cents -/
def box_cost : ℕ := 50

/-- The total volume of the collection in cubic inches -/
def collection_volume : ℕ := 3060000

/-- The number of boxes needed to package the collection -/
def boxes_needed : ℕ := (collection_volume + box_volume - 1) / box_volume

theorem minimum_cost_for_boxes : 
  boxes_needed * box_cost = 25500 :=
sorry

end minimum_cost_for_boxes_l1486_148665


namespace area_original_triangle_l1486_148600

/-- Given a triangle ABC and its oblique dimetric projection A''B''C'',
    where A''B''C'' is an equilateral triangle with side length a,
    prove that the area of ABC is (√6 * a^2) / 2. -/
theorem area_original_triangle (a : ℝ) (h : a > 0) :
  let s_projection := (Real.sqrt 3 * a^2) / 4
  let ratio := Real.sqrt 2 / 4
  s_projection / ratio = (Real.sqrt 6 * a^2) / 2 := by
sorry

end area_original_triangle_l1486_148600


namespace matrix_not_invertible_iff_l1486_148686

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2*x, 5],
    ![4*x, 9]]

theorem matrix_not_invertible_iff (x : ℝ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 0 := by sorry

end matrix_not_invertible_iff_l1486_148686


namespace smallest_two_digit_multiple_of_6_not_4_l1486_148620

theorem smallest_two_digit_multiple_of_6_not_4 :
  ∃ n : ℕ, 
    n ≥ 10 ∧ n < 100 ∧  -- two-digit positive integer
    n % 6 = 0 ∧         -- multiple of 6
    n % 4 ≠ 0 ∧         -- not a multiple of 4
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 6 = 0 ∧ m % 4 ≠ 0 → n ≤ m) ∧  -- smallest such number
    n = 18 :=           -- the number is 18
by sorry

end smallest_two_digit_multiple_of_6_not_4_l1486_148620


namespace areas_product_eq_volume_squared_l1486_148604

/-- A rectangular box with specific proportions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The volume of the box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- The area of the bottom of the box -/
def bottomArea (b : Box) : ℝ := b.length * b.width

/-- The area of the side of the box -/
def sideArea (b : Box) : ℝ := b.width * b.height

/-- The area of the front of the box -/
def frontArea (b : Box) : ℝ := b.length * b.height

/-- Theorem: The product of the areas equals the square of the volume -/
theorem areas_product_eq_volume_squared (b : Box) :
  bottomArea b * sideArea b * frontArea b = (volume b) ^ 2 := by
  sorry

end areas_product_eq_volume_squared_l1486_148604


namespace sum_of_numbers_with_given_difference_and_larger_l1486_148657

theorem sum_of_numbers_with_given_difference_and_larger (L S : ℤ) : 
  L = 35 → L - S = 15 → L + S = 55 := by sorry

end sum_of_numbers_with_given_difference_and_larger_l1486_148657


namespace senate_committee_seating_arrangements_l1486_148644

def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem senate_committee_seating_arrangements :
  circular_arrangements 10 = 362880 := by
  sorry

end senate_committee_seating_arrangements_l1486_148644


namespace factorization_of_expression_l1486_148658

theorem factorization_of_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (b^2 * c^3) := by sorry

end factorization_of_expression_l1486_148658


namespace distance_to_directrix_l1486_148679

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (h : p > 0) : 
  let A : ℝ × ℝ := (1, Real.sqrt 5)
  let C := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  A ∈ C → |1 - (-p/2)| = 9/4 := by
  sorry

end distance_to_directrix_l1486_148679


namespace cows_and_sheep_bushels_l1486_148629

/-- Represents the farm animals and their food consumption --/
structure Farm where
  cows : ℕ
  sheep : ℕ
  chickens : ℕ
  chicken_bushels : ℕ
  total_bushels : ℕ

/-- Calculates the bushels eaten by cows and sheep --/
def bushels_for_cows_and_sheep (farm : Farm) : ℕ :=
  farm.total_bushels - (farm.chickens * farm.chicken_bushels)

/-- Theorem stating that the bushels eaten by cows and sheep is 14 --/
theorem cows_and_sheep_bushels (farm : Farm) 
  (h1 : farm.cows = 4)
  (h2 : farm.sheep = 3)
  (h3 : farm.chickens = 7)
  (h4 : farm.chicken_bushels = 3)
  (h5 : farm.total_bushels = 35) :
  bushels_for_cows_and_sheep farm = 14 := by
  sorry

end cows_and_sheep_bushels_l1486_148629


namespace sequence_properties_l1486_148618

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the geometric sequence
def geometric_sequence (b : ℕ → ℝ) := ∀ n m, b (n + m) = b n * b m

theorem sequence_properties 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_cond : 2 * a 5 - a 3 = 3)
  (h_b_2 : b 2 = 1)
  (h_b_4 : b 4 = 4) :
  (a 7 = 3) ∧ 
  ((b 3 = 2 ∨ b 3 = -2) ∧ b 6 = 16) :=
sorry

end sequence_properties_l1486_148618


namespace complex_product_symmetric_imaginary_axis_l1486_148622

theorem complex_product_symmetric_imaginary_axis :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 2 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -5 := by
sorry

end complex_product_symmetric_imaginary_axis_l1486_148622


namespace correct_calculation_l1486_148661

theorem correct_calculation (x : ℤ) (h : x + 26 = 61) : x + 62 = 97 := by
  sorry

end correct_calculation_l1486_148661


namespace twin_prime_power_sum_divisibility_l1486_148663

theorem twin_prime_power_sum_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end twin_prime_power_sum_divisibility_l1486_148663


namespace odd_multiples_of_three_count_l1486_148611

theorem odd_multiples_of_three_count : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n % 3 = 0) (Finset.range 1001)).card = 167 := by
  sorry

end odd_multiples_of_three_count_l1486_148611


namespace horner_v3_eq_7_9_l1486_148694

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: Horner's method for f(x) at x = 1 gives v₃ = 7.9 -/
theorem horner_v3_eq_7_9 : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end horner_v3_eq_7_9_l1486_148694


namespace robie_cards_count_l1486_148677

theorem robie_cards_count (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) (boxes_remaining : ℕ) : 
  cards_per_box = 25 →
  unboxed_cards = 11 →
  boxes_given_away = 6 →
  boxes_remaining = 12 →
  cards_per_box * (boxes_given_away + boxes_remaining) + unboxed_cards = 461 := by
sorry

end robie_cards_count_l1486_148677


namespace fraction_simplification_l1486_148603

theorem fraction_simplification : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end fraction_simplification_l1486_148603


namespace counterexample_exists_l1486_148638

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l1486_148638


namespace smallest_n_with_common_factor_l1486_148623

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 19 → ¬(has_common_factor_greater_than_one (11*n - 3) (8*n + 2))) ∧ 
  (has_common_factor_greater_than_one (11*19 - 3) (8*19 + 2)) := by
  sorry

#check smallest_n_with_common_factor

end smallest_n_with_common_factor_l1486_148623


namespace pyramid_division_theorem_l1486_148624

/-- A structure representing a pyramid divided by planes parallel to its base -/
structure DividedPyramid (n : ℕ) where
  volumePlanes : Fin (n + 1) → ℝ
  surfacePlanes : Fin (n + 1) → ℝ

/-- The condition for a common plane between volume and surface divisions -/
def hasCommonPlane (n : ℕ) : Prop :=
  ∃ (i k : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k ≤ n ∧ (n + 1) * i^2 = k^3

/-- The list of n values up to 100 that satisfy the common plane condition -/
def validNValues : List ℕ :=
  [7, 15, 23, 26, 31, 39, 47, 53, 55, 63, 71, 79, 80, 87, 95]

/-- The condition for multiple common planes -/
def hasMultipleCommonPlanes (n : ℕ) : Prop :=
  ∃ (i₁ k₁ i₂ k₂ : ℕ),
    1 ≤ i₁ ∧ i₁ ≤ n ∧ 1 ≤ k₁ ∧ k₁ ≤ n ∧
    1 ≤ i₂ ∧ i₂ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
    (n + 1) * i₁^2 = k₁^3 ∧ (n + 1) * i₂^2 = k₂^3 ∧
    (i₁ ≠ i₂ ∨ k₁ ≠ k₂)

theorem pyramid_division_theorem :
  (∀ n ∈ validNValues, hasCommonPlane n) ∧
  (∀ n ∈ validNValues, n ≠ 63 → ¬hasMultipleCommonPlanes n) ∧
  hasMultipleCommonPlanes 63 :=
sorry

end pyramid_division_theorem_l1486_148624


namespace sport_formulation_water_amount_l1486_148610

/-- Represents the ratios and amounts in a flavored drink formulation -/
structure DrinkFormulation where
  standard_ratio_flavoring : ℚ
  standard_ratio_corn_syrup : ℚ
  standard_ratio_water : ℚ
  sport_ratio_flavoring_corn_syrup_multiplier : ℚ
  sport_ratio_flavoring_water_multiplier : ℚ
  sport_corn_syrup_amount : ℚ

/-- Calculates the amount of water in the sport formulation -/
def water_amount (d : DrinkFormulation) : ℚ :=
  let sport_ratio_flavoring := d.standard_ratio_flavoring
  let sport_ratio_corn_syrup := d.standard_ratio_corn_syrup / d.sport_ratio_flavoring_corn_syrup_multiplier
  let sport_ratio_water := d.standard_ratio_water / d.sport_ratio_flavoring_water_multiplier
  let flavoring_amount := d.sport_corn_syrup_amount * (sport_ratio_flavoring / sport_ratio_corn_syrup)
  flavoring_amount * (sport_ratio_water / sport_ratio_flavoring)

theorem sport_formulation_water_amount 
  (d : DrinkFormulation)
  (h1 : d.standard_ratio_flavoring = 1)
  (h2 : d.standard_ratio_corn_syrup = 12)
  (h3 : d.standard_ratio_water = 30)
  (h4 : d.sport_ratio_flavoring_corn_syrup_multiplier = 3)
  (h5 : d.sport_ratio_flavoring_water_multiplier = 2)
  (h6 : d.sport_corn_syrup_amount = 5) :
  water_amount d = 75/4 := by
  sorry

end sport_formulation_water_amount_l1486_148610


namespace smallest_resolvable_debt_is_correct_l1486_148680

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt that can be resolved using pigs and goats -/
def resolvable_debt (d : ℕ) : Prop :=
  ∃ (p g : ℤ), d = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (resolvable_debt smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(resolvable_debt d)) :=
sorry

end smallest_resolvable_debt_is_correct_l1486_148680


namespace boxwood_trim_charge_l1486_148613

/-- Calculates the total charge for trimming boxwoods with various shapes -/
def total_charge (basic_trim_cost sphere_cost pyramid_cost cube_cost : ℚ)
                 (total_boxwoods spheres pyramids cubes : ℕ) : ℚ :=
  basic_trim_cost * total_boxwoods +
  sphere_cost * spheres +
  pyramid_cost * pyramids +
  cube_cost * cubes

/-- Theorem stating the total charge for the given scenario -/
theorem boxwood_trim_charge :
  total_charge 5 15 20 25 30 4 3 2 = 320 := by
  sorry

end boxwood_trim_charge_l1486_148613


namespace shea_corn_purchase_l1486_148685

/-- The cost of corn per pound in cents -/
def corn_cost : ℕ := 110

/-- The cost of beans per pound in cents -/
def bean_cost : ℕ := 50

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℕ := 30

/-- The total cost in cents -/
def total_cost : ℕ := 2100

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 10

theorem shea_corn_purchase :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_pounds ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end shea_corn_purchase_l1486_148685


namespace amusement_park_ride_orders_l1486_148693

theorem amusement_park_ride_orders : Nat.factorial 6 = 720 := by
  sorry

end amusement_park_ride_orders_l1486_148693


namespace problem_solution_l1486_148666

theorem problem_solution : 9 - (3 / (1 / 3) + 3) = 3 := by
  sorry

end problem_solution_l1486_148666


namespace range_of_m_l1486_148650

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*m*x + 9 ≥ 0) → m ∈ Set.Icc (-2) 2 :=
by sorry

end range_of_m_l1486_148650


namespace optimal_plan_l1486_148674

/-- Represents the unit price of type A prizes -/
def price_A : ℝ := 30

/-- Represents the unit price of type B prizes -/
def price_B : ℝ := 15

/-- The total number of prizes to purchase -/
def total_prizes : ℕ := 30

/-- Condition: Total cost of 3 type A and 2 type B prizes is 120 yuan -/
axiom condition1 : 3 * price_A + 2 * price_B = 120

/-- Condition: Total cost of 5 type A and 4 type B prizes is 210 yuan -/
axiom condition2 : 5 * price_A + 4 * price_B = 210

/-- Function to calculate the total cost given the number of type A prizes -/
def total_cost (num_A : ℕ) : ℝ :=
  price_A * num_A + price_B * (total_prizes - num_A)

/-- Theorem stating the most cost-effective plan and its total cost -/
theorem optimal_plan :
  ∃ (num_A : ℕ),
    num_A ≥ (total_prizes - num_A) / 3 ∧
    num_A = 8 ∧
    total_cost num_A = 570 ∧
    ∀ (other_num_A : ℕ),
      other_num_A ≥ (total_prizes - other_num_A) / 3 →
      total_cost other_num_A ≥ total_cost num_A :=
sorry

end optimal_plan_l1486_148674


namespace officer_selection_count_l1486_148626

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers under given conditions --/
theorem officer_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 1584 := by
  sorry

#eval choose_officers 24 12 12

end officer_selection_count_l1486_148626


namespace inverse_proportion_min_value_l1486_148653

/-- Given an inverse proportion function y = k/x, prove that if the maximum value of y is 4
    when -2 ≤ x ≤ -1, then the minimum value of y is -1/2 when x ≥ 8 -/
theorem inverse_proportion_min_value (k : ℝ) :
  (∀ x, -2 ≤ x → x ≤ -1 → k / x ≤ 4) →
  (∃ x, -2 ≤ x ∧ x ≤ -1 ∧ k / x = 4) →
  (∀ x, x ≥ 8 → k / x ≥ -1/2) ∧
  (∃ x, x ≥ 8 ∧ k / x = -1/2) :=
by sorry

end inverse_proportion_min_value_l1486_148653


namespace sequence_squares_l1486_148630

theorem sequence_squares (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k^2
  (a 1 = 1) ∧ (a 2 = 4) ∧ (a 3 = 9) ∧ (a 4 = 16) ∧ (a 5 = 25) := by
  sorry

end sequence_squares_l1486_148630


namespace binomial_expansion_properties_l1486_148673

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (1+2x)^7 -/
def coefficient (r : ℕ) : ℕ := binomial 7 r * 2^r

theorem binomial_expansion_properties :
  (coefficient 2 = binomial 7 2 * 2^2) ∧
  (coefficient 2 = 24) := by sorry

end binomial_expansion_properties_l1486_148673


namespace length_of_AB_prime_l1486_148602

/-- Given points A, B, and C in the plane, with A' and B' on the line y = x,
    and lines AA' and BB' intersecting at C, prove that the length of A'B' is 120√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 5) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (∃ t : ℝ, A' = (1 - t) • A + t • C) →
  (∃ s : ℝ, B' = (1 - s) • B + s • C) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 120 * Real.sqrt 2 / 11 := by
  sorry

end length_of_AB_prime_l1486_148602


namespace modified_baseball_league_games_l1486_148664

/-- The total number of games played in a modified baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180 -/
theorem modified_baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

#eval total_games 10 4

end modified_baseball_league_games_l1486_148664


namespace expression_simplification_l1486_148647

theorem expression_simplification (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 3) (hr : r ≠ 4) : 
  (p - 2) / (4 - r) * (q - 3) / (2 - p) * (r - 4) / (3 - q) * (-2) = 2 := by
  sorry

end expression_simplification_l1486_148647


namespace is_center_of_hyperbola_l1486_148684

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 900 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((y - hyperbola_center.2)^2 / (819/36) - (x - hyperbola_center.1)^2 / (819/9) = 1) :=
sorry

end is_center_of_hyperbola_l1486_148684


namespace equal_angles_in_special_quadrilateral_l1486_148676

/-- A point on the Cartesian plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral on the Cartesian plane -/
structure Quadrilateral := (A B C D : Point)

/-- Checks if a point is on the hyperbola y = 1/x -/
def on_hyperbola (p : Point) : Prop := p.y = 1 / p.x

/-- Checks if a point is on the negative branch of the hyperbola -/
def on_negative_branch (p : Point) : Prop := on_hyperbola p ∧ p.x < 0

/-- Checks if a point is on the positive branch of the hyperbola -/
def on_positive_branch (p : Point) : Prop := on_hyperbola p ∧ p.x > 0

/-- Checks if a point is to the left of another point -/
def left_of (p1 p2 : Point) : Prop := p1.x < p2.x

/-- Checks if a line segment passes through the origin -/
def passes_through_origin (p1 p2 : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ t * p1.x + (1 - t) * p2.x = 0 ∧ t * p1.y + (1 - t) * p2.y = 0

/-- Calculates the angle between two lines given three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem equal_angles_in_special_quadrilateral (ABCD : Quadrilateral) :
  on_negative_branch ABCD.A →
  on_negative_branch ABCD.D →
  on_positive_branch ABCD.B →
  on_positive_branch ABCD.C →
  left_of ABCD.B ABCD.C →
  passes_through_origin ABCD.A ABCD.C →
  angle ABCD.B ABCD.A ABCD.D = angle ABCD.B ABCD.C ABCD.D :=
by sorry

end equal_angles_in_special_quadrilateral_l1486_148676


namespace first_class_equipment_amount_l1486_148616

/-- Represents the amount of equipment -/
structure Equipment where
  higherClass : ℕ
  firstClass : ℕ

/-- The initial distribution of equipment at two sites -/
structure InitialDistribution where
  site1 : Equipment
  site2 : Equipment

/-- The final distribution of equipment after transfers -/
structure FinalDistribution where
  site1 : Equipment
  site2 : Equipment

/-- Transfers equipment between sites according to the problem description -/
def transfer (init : InitialDistribution) : FinalDistribution :=
  sorry

/-- The conditions of the problem -/
def problemConditions (init : InitialDistribution) (final : FinalDistribution) : Prop :=
  init.site1.firstClass = 0 ∧
  init.site2.higherClass = 0 ∧
  init.site1.higherClass < init.site2.firstClass ∧
  final = transfer init ∧
  final.site1.higherClass = final.site2.higherClass + 26 ∧
  final.site2.higherClass + final.site2.firstClass > 
    (init.site2.higherClass + init.site2.firstClass) * 21 / 20

theorem first_class_equipment_amount 
  (init : InitialDistribution) 
  (final : FinalDistribution) 
  (h : problemConditions init final) : 
  init.site2.firstClass = 60 :=
sorry

end first_class_equipment_amount_l1486_148616


namespace visible_black_area_ratio_l1486_148687

/-- Represents the area of a circle -/
structure CircleArea where
  area : ℝ
  area_pos : area > 0

/-- Represents the configuration of three circles -/
structure CircleConfiguration where
  black : CircleArea
  grey : CircleArea
  white : CircleArea
  initial_visible_black : ℝ
  final_visible_black : ℝ
  initial_condition : initial_visible_black = 7 * white.area
  final_condition : final_visible_black = initial_visible_black - white.area

/-- The theorem stating the ratio of visible black areas before and after rearrangement -/
theorem visible_black_area_ratio (config : CircleConfiguration) :
  config.initial_visible_black / config.final_visible_black = 7 / 6 := by
  sorry

end visible_black_area_ratio_l1486_148687


namespace min_value_I_l1486_148659

theorem min_value_I (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x ≥ 0) (hy : y ≥ 0)
  (h_sum : a^6 + b^6 + c^6 = 3)
  (h_constraint : (x + 1)^2 + y^2 ≤ 2) : 
  let I := 1 / (2*a^3*x + b^3*y^2) + 1 / (2*b^3*x + c^3*y^2) + 1 / (2*c^3*x + a^3*y^2)
  ∀ I', I ≥ I' → I' ≥ 3 :=
sorry

end min_value_I_l1486_148659


namespace min_value_inequality_min_value_achievable_l1486_148654

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (a b c d : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 = 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

end min_value_inequality_min_value_achievable_l1486_148654


namespace fixed_point_on_line_l1486_148634

/-- The line equation ax + (2a-1)y + a-3 = 0 passes through the point (5, -3) for all values of a. -/
theorem fixed_point_on_line (a : ℝ) : a * 5 + (2 * a - 1) * (-3) + a - 3 = 0 := by
  sorry

#check fixed_point_on_line

end fixed_point_on_line_l1486_148634


namespace distinct_cube_edge_colorings_l1486_148670

/-- The group of rotations of the cube -/
structure CubeRotationGroup where
  D : Type
  mul : D → D → D

/-- The permutation group of the edges of the cube induced by the rotation group -/
structure EdgePermutationGroup where
  W : Type
  comp : W → W → W

/-- The cycle index polynomial for the permutation group (W, ∘) -/
def cycle_index_polynomial (W : EdgePermutationGroup) : ℕ :=
  sorry

/-- The number of distinct colorings for a given permutation type -/
def colorings_for_permutation (perm_type : String) : ℕ :=
  sorry

/-- Theorem: The number of distinct ways to color the edges of a cube with 3 red, 3 blue, and 6 yellow edges is 780 -/
theorem distinct_cube_edge_colorings :
  let num_edges : ℕ := 12
  let num_red : ℕ := 3
  let num_blue : ℕ := 3
  let num_yellow : ℕ := 6
  (num_red + num_blue + num_yellow = num_edges) →
  (∃ (W : EdgePermutationGroup),
    (cycle_index_polynomial W *
     (colorings_for_permutation "1^12" +
      8 * colorings_for_permutation "3^4" +
      6 * colorings_for_permutation "1^2 2^5")) / 24 = 780) :=
by
  sorry

end distinct_cube_edge_colorings_l1486_148670


namespace horners_method_operations_l1486_148633

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Horner's method representation of the polynomial -/
def horner_f (x : ℝ) : ℝ := ((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1

/-- The number of multiplication operations in Horner's method for this polynomial -/
def mult_ops : ℕ := 5

/-- The number of addition operations in Horner's method for this polynomial -/
def add_ops : ℕ := 5

theorem horners_method_operations :
  f 5 = horner_f 5 ∧ mult_ops = 5 ∧ add_ops = 5 := by sorry

end horners_method_operations_l1486_148633


namespace max_product_digits_sum_23_l1486_148646

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The product of digits of a positive integer -/
def product_of_digits (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum product of digits for a positive integer with digit sum 23 is 432 -/
theorem max_product_digits_sum_23 :
  ∀ n : ℕ+, sum_of_digits n = 23 → product_of_digits n ≤ 432 :=
sorry

end max_product_digits_sum_23_l1486_148646


namespace total_savings_ten_sets_l1486_148617

/-- The cost of 2 packs of milk -/
def cost_two_packs : ℚ := 2.50

/-- The cost of an individual pack of milk -/
def cost_individual : ℚ := 1.30

/-- The number of sets being purchased -/
def num_sets : ℕ := 10

/-- The number of packs in each set -/
def packs_per_set : ℕ := 2

/-- Theorem stating the total savings from buying ten sets of 2 packs of milk -/
theorem total_savings_ten_sets : 
  (num_sets * packs_per_set) * (cost_individual - cost_two_packs / 2) = 1 := by
  sorry

end total_savings_ten_sets_l1486_148617


namespace limit_ratio_sevens_to_total_l1486_148643

/-- Count of digit 7 occurrences in decimal representation of numbers from 1 to n -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Total count of digits in decimal representation of numbers from 1 to n -/
def total_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the limit of the ratio of 7's to total digits is 1/10 -/
theorem limit_ratio_sevens_to_total (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n ≥ N, |((count_sevens n : ℝ) / (total_digits n : ℝ)) - (1 / 10)| < ε :=
sorry

end limit_ratio_sevens_to_total_l1486_148643


namespace modulus_of_complex_fraction_l1486_148637

theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + Complex.I * Real.sqrt 3) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end modulus_of_complex_fraction_l1486_148637


namespace pentagonal_base_monochromatic_l1486_148636

-- Define the vertices of the prism
inductive Vertex : Type
| A : Fin 5 → Vertex
| B : Fin 5 → Vertex

-- Define the color of an edge
inductive Color : Type
| Red
| Blue

-- Define the edge coloring function
def edge_color : Vertex → Vertex → Color := sorry

-- No triangle has all edges of the same color
axiom no_monochromatic_triangle :
  ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 → v2 ≠ v3 → v3 ≠ v1 →
    ¬(edge_color v1 v2 = edge_color v2 v3 ∧ edge_color v2 v3 = edge_color v3 v1)

-- Theorem: All edges of each pentagonal base are the same color
theorem pentagonal_base_monochromatic :
  (∀ (i j : Fin 5), edge_color (Vertex.A i) (Vertex.A j) = edge_color (Vertex.A 0) (Vertex.A 1)) ∧
  (∀ (i j : Fin 5), edge_color (Vertex.B i) (Vertex.B j) = edge_color (Vertex.B 0) (Vertex.B 1)) :=
sorry

end pentagonal_base_monochromatic_l1486_148636


namespace find_x_value_l1486_148696

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 0} →
  B = {0, 1, x + 2} →
  A ⊆ B →
  x = -3 := by
sorry

end find_x_value_l1486_148696


namespace polynomial_sum_independence_l1486_148662

theorem polynomial_sum_independence (a b : ℝ) :
  (∀ x y : ℝ, (x^2 + a*x - y + b) + (b*x^2 - 3*x + 6*y - 3) = (5*y + b - 3)) →
  3*(a^2 - 2*a*b + b^2) - (4*a^2 - 2*(1/2*a^2 + a*b - 3/2*b^2)) = 12 := by
sorry

end polynomial_sum_independence_l1486_148662


namespace max_distance_complex_l1486_148656

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ z, |(2 + 3*I)*z^2 - z^4|) = 81 + 9 * Real.sqrt 13 :=
sorry

end max_distance_complex_l1486_148656


namespace a_minus_b_equals_two_l1486_148607

theorem a_minus_b_equals_two (a b : ℝ) 
  (h1 : |a| = 1) 
  (h2 : |b - 1| = 2) 
  (h3 : a > b) : 
  a - b = 2 := by
sorry

end a_minus_b_equals_two_l1486_148607


namespace total_study_hours_l1486_148691

/-- The number of weeks in the fall semester -/
def semester_weeks : ℕ := 15

/-- The number of study hours on weekdays -/
def weekday_hours : ℕ := 3

/-- The number of study hours on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of study hours on Sunday -/
def sunday_hours : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Theorem stating the total study hours during the semester -/
theorem total_study_hours :
  semester_weeks * (weekdays_per_week * weekday_hours + saturday_hours + sunday_hours) = 360 := by
  sorry

end total_study_hours_l1486_148691


namespace problem_solution_l1486_148625

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) 
  (h_solution_set : {x : ℝ | f m (x - 3) ≥ 0} = Set.Iic (-2) ∪ Set.Ici 2) :
  m = 2 ∧ 
  ∀ (x t : ℝ), f 2 x ≥ |2 * x - 1| - t^2 + (3/2) * t + 1 → 
    t ∈ Set.Iic (1/2) ∪ Set.Ici 1 := by
sorry

end problem_solution_l1486_148625


namespace solution_set_f_gt_5_range_of_a_empty_solution_l1486_148631

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for part I
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -4/3 ∨ x > 2} :=
sorry

-- Theorem for part II
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, 1 / (f x - 4) ≠ a} = {a : ℝ | -2/3 < a ∧ a ≤ 0} :=
sorry

end solution_set_f_gt_5_range_of_a_empty_solution_l1486_148631


namespace lawrence_county_kids_count_l1486_148655

theorem lawrence_county_kids_count :
  let kids_stayed_home : ℕ := 644997
  let kids_went_to_camp : ℕ := 893835
  let outside_kids_at_camp : ℕ := 78
  kids_stayed_home + kids_went_to_camp = 1538832 :=
by sorry

end lawrence_county_kids_count_l1486_148655


namespace range_of_m_l1486_148642

theorem range_of_m (m : ℝ) : m ≥ 3 ↔ 
  (∀ x : ℝ, (|2*x + 1| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, |2*x + 1| > 3 ∧ x^2 - 2*x + 1 - m^2 > 0)) ∧ 
  m > 0 :=
by sorry

end range_of_m_l1486_148642


namespace reflection_squared_is_identity_l1486_148671

/-- A reflection matrix over a non-zero vector -/
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- The identity matrix -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = identity_matrix :=
sorry

end reflection_squared_is_identity_l1486_148671


namespace aquarium_visitors_l1486_148639

/-- Calculates the number of healthy visitors given total visitors and ill percentage --/
def healthyVisitors (total : ℕ) (illPercentage : ℕ) : ℕ :=
  total - (total * illPercentage) / 100

/-- Properties of the aquarium visits over three days --/
theorem aquarium_visitors :
  let mondayTotal := 300
  let mondayIllPercentage := 15
  let tuesdayTotal := 500
  let tuesdayIllPercentage := 30
  let wednesdayTotal := 400
  let wednesdayIllPercentage := 20
  
  (healthyVisitors mondayTotal mondayIllPercentage +
   healthyVisitors tuesdayTotal tuesdayIllPercentage +
   healthyVisitors wednesdayTotal wednesdayIllPercentage) = 925 := by
  sorry

end aquarium_visitors_l1486_148639


namespace work_completion_time_work_completion_result_l1486_148689

/-- The time taken to complete a work when two people work together, given their individual completion times -/
theorem work_completion_time (ajay_time vijay_time : ℝ) (h1 : ajay_time > 0) (h2 : vijay_time > 0) :
  (ajay_time * vijay_time) / (ajay_time + vijay_time) = 
    (8 : ℝ) * 24 / ((8 : ℝ) + 24) :=
by sorry

/-- The result of the work completion time calculation is 6 days -/
theorem work_completion_result :
  (8 : ℝ) * 24 / ((8 : ℝ) + 24) = 6 :=
by sorry

end work_completion_time_work_completion_result_l1486_148689


namespace school_average_gpa_l1486_148615

theorem school_average_gpa (gpa_6th : ℝ) (gpa_7th : ℝ) (gpa_8th : ℝ)
  (h1 : gpa_6th = 93)
  (h2 : gpa_7th = gpa_6th + 2)
  (h3 : gpa_8th = 91) :
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 := by
  sorry

end school_average_gpa_l1486_148615


namespace value_of_expression_l1486_148605

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end value_of_expression_l1486_148605


namespace sum_of_digits_M_l1486_148683

/-- The sum of the first n 9's (e.g., 9, 99, 999, ...) -/
def sumOfNines (n : ℕ) : ℕ := (10^n - 1)

/-- The sum of the digits of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- M is defined as the sum of the first five 9's -/
def M : ℕ := (sumOfNines 1) + (sumOfNines 2) + (sumOfNines 3) + (sumOfNines 4) + (sumOfNines 5)

theorem sum_of_digits_M : digitSum M = 8 := by
  sorry

end sum_of_digits_M_l1486_148683


namespace exists_shorter_representation_l1486_148635

def repeatedSevens (n : ℕ) : ℕ := 
  (7 * (10^n - 1)) / 9

def validExpression (expr : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ), expr k = repeatedSevens k ∧ 
  (∀ m : ℕ, m ≤ k → expr m ≠ repeatedSevens m)

theorem exists_shorter_representation : 
  ∃ (n : ℕ) (expr : ℕ → ℕ), n > 2 ∧ validExpression expr ∧ 
  (∀ k : ℕ, k ≥ n → expr k < repeatedSevens k) :=
sorry

end exists_shorter_representation_l1486_148635


namespace quadratic_roots_sum_l1486_148672

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → 
  (b^2 + 8*b + 4 = 0) → 
  (a ≠ 0) →
  (b ≠ 0) →
  (a/b + b/a = 14) :=
by
  sorry

end quadratic_roots_sum_l1486_148672


namespace smallest_x_value_l1486_148660

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (254 + x)) : 
  x ≥ 2 ∧ ∃ (y' : ℕ+), (3 : ℚ) / 4 = y' / (254 + 2) :=
sorry

end smallest_x_value_l1486_148660


namespace carbon_processing_optimization_l1486_148612

-- Define the processing volume range
def ProcessingRange : Set ℝ := {x : ℝ | 300 ≤ x ∧ x ≤ 600}

-- Define the cost function
def CostFunction (x : ℝ) : ℝ := 0.5 * x^2 - 200 * x + 45000

-- Define the revenue function
def RevenueFunction (x : ℝ) : ℝ := 200 * x

-- Define the profit function
def ProfitFunction (x : ℝ) : ℝ := RevenueFunction x - CostFunction x

-- Theorem statement
theorem carbon_processing_optimization :
  ∃ (x_min : ℝ) (max_profit : ℝ),
    x_min ∈ ProcessingRange ∧
    (∀ x ∈ ProcessingRange, CostFunction x_min / x_min ≤ CostFunction x / x) ∧
    x_min = 300 ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x > 0) ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x ≤ max_profit) ∧
    max_profit = 35000 := by
  sorry

end carbon_processing_optimization_l1486_148612


namespace geometry_textbook_weight_l1486_148601

theorem geometry_textbook_weight 
  (chemistry_weight : Real) 
  (weight_difference : Real) 
  (h1 : chemistry_weight = 7.12)
  (h2 : weight_difference = 6.5)
  (h3 : chemistry_weight = geometry_weight + weight_difference) :
  geometry_weight = 0.62 :=
by
  sorry

end geometry_textbook_weight_l1486_148601


namespace arithmetic_geometric_mean_sum_squares_l1486_148645

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 80) : 
  x^2 + y^2 = 1440 := by
sorry

end arithmetic_geometric_mean_sum_squares_l1486_148645


namespace highway_vehicles_l1486_148675

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 40

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 800

/-- The number of vehicles per 100 million for the accident rate calculation -/
def base_vehicles : ℕ := 100000000

/-- The number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

theorem highway_vehicles :
  total_vehicles = (total_accidents * base_vehicles) / accident_rate :=
sorry

end highway_vehicles_l1486_148675


namespace larger_segment_approx_59_l1486_148667

/-- Triangle with sides 40, 50, and 110 units --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 40
  hb : b = 50
  hc : c = 110

/-- Altitude dropped on the longest side --/
def altitude (t : Triangle) : ℝ := sorry

/-- Larger segment cut off on the longest side --/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is approximately 59 units --/
theorem larger_segment_approx_59 (t : Triangle) :
  |larger_segment t - 59| < 0.5 := by sorry

end larger_segment_approx_59_l1486_148667


namespace smallest_n_inequality_l1486_148698

theorem smallest_n_inequality (w x y z : ℝ) : 
  ∃ (n : ℕ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n*(w^4 + x^4 + y^4 + z^4) ∧ 
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4) :=
by
  sorry

end smallest_n_inequality_l1486_148698


namespace jerry_read_30_pages_saturday_l1486_148627

/-- The number of pages Jerry read on Saturday -/
def pages_read_saturday (total_pages : ℕ) (pages_read_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  total_pages - (pages_remaining + pages_read_sunday)

theorem jerry_read_30_pages_saturday :
  pages_read_saturday 93 20 43 = 30 := by
  sorry

end jerry_read_30_pages_saturday_l1486_148627


namespace tan_equality_345_degrees_l1486_148651

theorem tan_equality_345_degrees (n : ℤ) :
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end tan_equality_345_degrees_l1486_148651


namespace arithmetic_sequence_sum_l1486_148699

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating that if S_2 = 3 and S_3 = 3, then S_5 = 0 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h2 : seq.S 2 = 3) (h3 : seq.S 3 = 3) : seq.S 5 = 0 := by
  sorry

end arithmetic_sequence_sum_l1486_148699


namespace stating_largest_valid_n_l1486_148621

/-- 
Given a positive integer n, this function checks if n! can be expressed as 
the product of n - 4 consecutive positive integers.
-/
def is_valid (n : ℕ) : Prop :=
  ∃ (b : ℕ), b ≥ 4 ∧ n.factorial = ((n - 4 + b).factorial / b.factorial)

/-- 
Theorem stating that 119 is the largest positive integer n for which n! 
can be expressed as the product of n - 4 consecutive positive integers.
-/
theorem largest_valid_n : 
  (is_valid 119) ∧ (∀ m : ℕ, m > 119 → ¬(is_valid m)) :=
sorry

end stating_largest_valid_n_l1486_148621


namespace zeros_not_adjacent_probability_l1486_148608

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 := by sorry

end zeros_not_adjacent_probability_l1486_148608


namespace function_periodicity_l1486_148609

/-- A function satisfying the given functional equation is periodic with period 4 -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 4) = f x := by
  sorry

end function_periodicity_l1486_148609


namespace special_functions_bound_l1486_148695

open Real

/-- Two differentiable real functions satisfying the given conditions -/
structure SpecialFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  hf : Differentiable ℝ f
  hg : Differentiable ℝ g
  h_eq : ∀ x, deriv f x / deriv g x = exp (f x - g x)
  h_f0 : f 0 = 1
  h_g2003 : g 2003 = 1

/-- The theorem stating that f(2003) > 1 - ln(2) for any pair of functions satisfying the conditions,
    and that 1 - ln(2) is the largest such constant -/
theorem special_functions_bound (sf : SpecialFunctions) :
  sf.f 2003 > 1 - log 2 ∧ ∀ c, (∀ sf' : SpecialFunctions, sf'.f 2003 > c) → c ≤ 1 - log 2 := by
  sorry

end special_functions_bound_l1486_148695


namespace quadratic_through_origin_l1486_148619

/-- Given a quadratic function f(x) = ax^2 + x + a(a-2) that passes through the origin,
    prove that a = 2 -/
theorem quadratic_through_origin (a : ℝ) (h1 : a ≠ 0) :
  (∀ x, a*x^2 + x + a*(a-2) = 0 → x = 0) → a = 2 := by
  sorry

end quadratic_through_origin_l1486_148619


namespace two_hour_walk_distance_l1486_148678

/-- Calculates the total distance walked in two hours given the distance walked in the first hour -/
def total_distance (first_hour_distance : ℝ) : ℝ :=
  first_hour_distance + 2 * first_hour_distance

/-- Theorem stating that walking 2 km in the first hour and twice that in the second hour results in 6 km total -/
theorem two_hour_walk_distance :
  total_distance 2 = 6 := by
  sorry

end two_hour_walk_distance_l1486_148678


namespace determinant_equals_t_minus_s_plus_r_l1486_148697

-- Define the polynomial
def polynomial (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![1+a, 1,   1,   1],
    ![1,   1+b, 1,   1],
    ![1,   1,   1+c, 1],
    ![1,   1,   1,   1+d]]

theorem determinant_equals_t_minus_s_plus_r 
  (r s t : ℝ) (a b c d : ℝ) 
  (h1 : polynomial a r s t = 0)
  (h2 : polynomial b r s t = 0)
  (h3 : polynomial c r s t = 0)
  (h4 : polynomial d r s t = 0) :
  Matrix.det (matrix a b c d) = t - s + r := by
  sorry

end determinant_equals_t_minus_s_plus_r_l1486_148697
