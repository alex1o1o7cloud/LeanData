import Mathlib

namespace NUMINAMATH_CALUDE_inverse_113_mod_114_l1566_156678

theorem inverse_113_mod_114 : ∃ x : ℕ, x ≡ 113 [ZMOD 114] ∧ 113 * x ≡ 1 [ZMOD 114] :=
by sorry

end NUMINAMATH_CALUDE_inverse_113_mod_114_l1566_156678


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l1566_156617

/-- Proves that the number of mystery book shelves is 8 --/
theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 7
  let picture_book_shelves : ℕ := 2
  let total_books : ℕ := 70
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 8 := by
sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l1566_156617


namespace NUMINAMATH_CALUDE_helper_sequences_count_l1566_156620

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of student helpers possible in a week -/
def helper_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating that the number of different sequences of student helpers in a week is 3375 -/
theorem helper_sequences_count : helper_sequences = 3375 := by
  sorry

end NUMINAMATH_CALUDE_helper_sequences_count_l1566_156620


namespace NUMINAMATH_CALUDE_dealer_profit_is_sixty_percent_l1566_156670

/-- Calculates the dealer's profit percentage given the purchase and sale information. -/
def dealer_profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_price_per_article := purchase_price / purchase_quantity
  let selling_price_per_article := sale_price / sale_quantity
  let profit_per_article := selling_price_per_article - cost_price_per_article
  (profit_per_article / cost_price_per_article) * 100

/-- Theorem stating that the dealer's profit percentage is 60% given the specified conditions. -/
theorem dealer_profit_is_sixty_percent :
  dealer_profit_percentage 15 25 12 32 = 60 := by
  sorry

end NUMINAMATH_CALUDE_dealer_profit_is_sixty_percent_l1566_156670


namespace NUMINAMATH_CALUDE_max_sum_nonnegative_reals_l1566_156632

theorem max_sum_nonnegative_reals (a b c : ℝ) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  a^2 + b^2 + c^2 = 52 →
  a*b + b*c + c*a = 28 →
  a + b + c ≤ 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_nonnegative_reals_l1566_156632


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1566_156602

/-- Given a train of length 110 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds,
    prove that the length of the bridge is 265 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 265 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1566_156602


namespace NUMINAMATH_CALUDE_unique_parallel_line_l1566_156696

-- Define a type for points in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a type for lines in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define what it means for a point to be on a line
def PointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be parallel
def ParallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- State the theorem
theorem unique_parallel_line (L : Line) (P : Point) 
  (h : ¬ PointOnLine P L) : 
  ∃! M : Line, ParallelLines M L ∧ PointOnLine P M :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l1566_156696


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1566_156625

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Theorem: In a trapezoid with given properties, the length of the other side is 10 -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h_area : t.area = 164)
  (h_altitude : t.altitude = 8)
  (h_base1 : t.base1 = 10)
  (h_base2 : t.base2 = 17) :
  t.base2 - t.base1 = 10 := by
  sorry

#check trapezoid_side_length

end NUMINAMATH_CALUDE_trapezoid_side_length_l1566_156625


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1566_156688

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total number of employees -/
def totalEmployees (ec : EmployeeCount) : Nat :=
  ec.young + ec.middleAged + ec.elderly

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total sample size -/
def totalSampleSize (ss : SampleSize) : Nat :=
  ss.young + ss.middleAged + ss.elderly

theorem stratified_sampling_theorem (ec : EmployeeCount) (ss : SampleSize) :
  totalEmployees ec = 750 ∧
  ec.young = 350 ∧
  ec.middleAged = 250 ∧
  ec.elderly = 150 ∧
  ss.young = 7 →
  totalSampleSize ss = 15 :=
sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1566_156688


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1566_156685

theorem absolute_value_equation (x y : ℝ) : 
  |x^2 - Real.log y| = x^2 + Real.log y → x * (y - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1566_156685


namespace NUMINAMATH_CALUDE_chestnut_collection_l1566_156649

/-- The chestnut collection problem -/
theorem chestnut_collection
  (a b c : ℝ)
  (mean_ab_c : (a + b) / 2 = c + 10)
  (mean_ac_b : (a + c) / 2 = b - 3) :
  (b + c) / 2 - a = -7 := by
  sorry

end NUMINAMATH_CALUDE_chestnut_collection_l1566_156649


namespace NUMINAMATH_CALUDE_triangle_yz_length_l1566_156642

/-- Given a triangle XYZ where cos(2X-Z) + sin(X+Y) = 2 and XY = 6, prove that YZ = 6√2 -/
theorem triangle_yz_length (X Y Z : Real) (h1 : Real.cos (2*X - Z) + Real.sin (X + Y) = 2) 
  (h2 : 0 < X ∧ X < π) (h3 : 0 < Y ∧ Y < π) (h4 : 0 < Z ∧ Z < π) 
  (h5 : X + Y + Z = π) (h6 : XY = 6) : 
  let YZ := Real.sqrt ((XY^2) * 2)
  YZ = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_yz_length_l1566_156642


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l1566_156697

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℝ
  day : Nat

/-- Represents the volume of orangeade made on a given day -/
structure OrangeadeVolume where
  volume : ℝ
  day : Nat

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : OrangeadeVolume) : ℝ :=
  price.price * volume.volume

theorem orangeade_price_day2 
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice)
  (volume_day1 : OrangeadeVolume)
  (volume_day2 : OrangeadeVolume)
  (h1 : price_day1.day = 1)
  (h2 : price_day2.day = 2)
  (h3 : volume_day1.day = 1)
  (h4 : volume_day2.day = 2)
  (h5 : price_day1.price = 0.82)
  (h6 : volume_day2.volume = (3/2) * volume_day1.volume)
  (h7 : revenue price_day1 volume_day1 = revenue price_day2 volume_day2) :
  price_day2.price = (2 * 0.82) / 3 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l1566_156697


namespace NUMINAMATH_CALUDE_circle_area_l1566_156674

theorem circle_area (circumference : ℝ) (area : ℝ) : 
  circumference = 36 → area = 324 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1566_156674


namespace NUMINAMATH_CALUDE_congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l1566_156669

-- Define geometric figures
structure GeometricFigure where
  -- Add necessary properties for a geometric figure
  -- This is a simplified representation
  shape : ℕ
  size : ℝ

-- Define congruence relation
def congruent (a b : GeometricFigure) : Prop :=
  a.shape = b.shape ∧ a.size = b.size

-- Define similarity relation with scale factor
def similar (a b : GeometricFigure) (scale : ℝ) : Prop :=
  a.shape = b.shape ∧ a.size = scale * b.size

-- Theorem: Congruent figures are similar with scale factor 1
theorem congruent_implies_similar (a b : GeometricFigure) :
  congruent a b → similar a b 1 := by
  sorry

-- Theorem: Similar figures with scale factor 1 are congruent
theorem similar_scale_one_implies_congruent (a b : GeometricFigure) :
  similar a b 1 → congruent a b := by
  sorry

-- Theorem: Congruent figures are a subset of similar figures
theorem congruent_subset_similar (a b : GeometricFigure) :
  congruent a b → ∃ scale, similar a b scale := by
  sorry

end NUMINAMATH_CALUDE_congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l1566_156669


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l1566_156628

theorem smallest_integer_inequality (x y z w : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m * (a^4 + b^4 + c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l1566_156628


namespace NUMINAMATH_CALUDE_extra_crayons_l1566_156661

theorem extra_crayons (packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  packs = 4 → crayons_per_pack = 10 → total_crayons = 46 → 
  total_crayons - (packs * crayons_per_pack) = 6 := by
  sorry

end NUMINAMATH_CALUDE_extra_crayons_l1566_156661


namespace NUMINAMATH_CALUDE_max_constant_quadratic_real_roots_l1566_156600

theorem max_constant_quadratic_real_roots :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 6*x + c = 0) → c ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_constant_quadratic_real_roots_l1566_156600


namespace NUMINAMATH_CALUDE_expression_simplification_l1566_156616

theorem expression_simplification (x : ℤ) (h : x = 2018) :
  x^2 + 2*x - x*(x + 1) = 2018 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1566_156616


namespace NUMINAMATH_CALUDE_quadratic_sum_l1566_156635

/-- A quadratic function f(x) = ax^2 + bx + c with a minimum value of 36
    and roots at x = 1 and x = 5 has the property that a + b + c = 0 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≥ 36) ∧ 
  (∃ x₀, ∀ x, a * x^2 + b * x + c ≥ a * x₀^2 + b * x₀ + c ∧ a * x₀^2 + b * x₀ + c = 36) ∧
  (a * 1^2 + b * 1 + c = 0) ∧
  (a * 5^2 + b * 5 + c = 0) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1566_156635


namespace NUMINAMATH_CALUDE_inequality_proof_l1566_156653

/-- If for all real x, 1 - a cos x - b sin x - A cos 2x - B sin 2x ≥ 0, 
    then a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem inequality_proof (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1566_156653


namespace NUMINAMATH_CALUDE_units_digit_of_M_M7_l1566_156657

def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => 2 * M (n + 1) + M n

theorem units_digit_of_M_M7 : M (M 7) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M7_l1566_156657


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1566_156647

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), (433124 + m) % 17 = 0 → m ≥ n) ∧ 
  (433124 + n) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1566_156647


namespace NUMINAMATH_CALUDE_two_digit_product_digits_l1566_156683

theorem two_digit_product_digits :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (100 ≤ a * b ∧ a * b ≤ 9999) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_digits_l1566_156683


namespace NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l1566_156693

/-- Represents the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of pourings required to have exactly 1/5 of the original water remaining -/
def requiredPourings : ℕ := 8

theorem water_remaining_after_required_pourings :
  waterRemaining requiredPourings = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l1566_156693


namespace NUMINAMATH_CALUDE_specific_prism_surface_area_l1566_156659

/-- A right triangular prism with given dimensions -/
structure RightTriangularPrism where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  height : ℝ

/-- Calculate the surface area of a right triangular prism -/
def surfaceArea (prism : RightTriangularPrism) : ℝ :=
  prism.leg1 * prism.leg2 + (prism.leg1 + prism.leg2 + prism.hypotenuse) * prism.height

/-- The surface area of the specific right triangular prism is 72 -/
theorem specific_prism_surface_area :
  let prism : RightTriangularPrism := {
    leg1 := 3,
    leg2 := 4,
    hypotenuse := 5,
    height := 5
  }
  surfaceArea prism = 72 := by sorry

end NUMINAMATH_CALUDE_specific_prism_surface_area_l1566_156659


namespace NUMINAMATH_CALUDE_cycle_selling_price_l1566_156641

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 1400 →
  loss_percentage = 15 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  selling_price = 1190 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l1566_156641


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1566_156643

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + n - 1 = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ m * x + y + n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1566_156643


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1566_156681

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 - 8*x - 1 ↔ y = (x - 4)^2 - 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1566_156681


namespace NUMINAMATH_CALUDE_correct_side_for_significant_figures_l1566_156634

/-- Represents the side from which we start counting significant figures -/
inductive Side
  | Left
  | Right
  | Front
  | Back

/-- Definition of significant figures for an approximate number -/
def significantFigures (number : ℕ) (startSide : Side) : ℕ :=
  sorry

/-- Theorem stating that the correct side to start from for significant figures is the left side -/
theorem correct_side_for_significant_figures :
  ∀ (number : ℕ), significantFigures number Side.Left = significantFigures number Side.Left :=
  sorry

end NUMINAMATH_CALUDE_correct_side_for_significant_figures_l1566_156634


namespace NUMINAMATH_CALUDE_tv_contest_probabilities_l1566_156677

-- Define the pass rates for each level
def pass_rate_1 : ℝ := 0.6
def pass_rate_2 : ℝ := 0.5
def pass_rate_3 : ℝ := 0.4

-- Define the prize amounts
def first_prize : ℕ := 300
def second_prize : ℕ := 200

-- Define the function to calculate the probability of not winning any prize
def prob_no_prize : ℝ := 1 - pass_rate_1 + pass_rate_1 * (1 - pass_rate_2)

-- Define the function to calculate the probability of total prize money being 700,
-- given both contestants passed the first level
def prob_total_700_given_pass_1 : ℝ :=
  2 * (pass_rate_2 * (1 - pass_rate_3)) * (pass_rate_2 * pass_rate_3)

-- State the theorem
theorem tv_contest_probabilities :
  prob_no_prize = 0.7 ∧
  prob_total_700_given_pass_1 = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_tv_contest_probabilities_l1566_156677


namespace NUMINAMATH_CALUDE_probability_no_adjacent_seating_l1566_156680

def num_chairs : ℕ := 9
def num_people : ℕ := 4

def total_arrangements (n m : ℕ) : ℕ :=
  (n - 1) * (n - 2) * (n - 3)

def favorable_arrangements (n m : ℕ) : ℕ :=
  (n - m + 1) * m

theorem probability_no_adjacent_seating :
  (favorable_arrangements num_chairs num_people : ℚ) / 
  (total_arrangements num_chairs num_people : ℚ) = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_seating_l1566_156680


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1566_156626

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1566_156626


namespace NUMINAMATH_CALUDE_remove_number_for_average_l1566_156605

theorem remove_number_for_average (list : List ℕ) (removed : ℕ) (avg : ℚ) : 
  list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] →
  removed = 6 →
  avg = 82/10 →
  (list.sum - removed) / (list.length - 1) = avg := by
  sorry

end NUMINAMATH_CALUDE_remove_number_for_average_l1566_156605


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l1566_156603

/-- A quadrilateral with vertices at (1,2), (4,6), (6,5), and (4,1) -/
def Quadrilateral : Set (ℝ × ℝ) :=
  {(1, 2), (4, 6), (6, 5), (4, 1)}

/-- The perimeter of the quadrilateral -/
noncomputable def perimeter : ℝ :=
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist (1, 2) (4, 6) + dist (4, 6) (6, 5) + dist (6, 5) (4, 1) + dist (4, 1) (1, 2)

/-- The theorem to be proved -/
theorem quadrilateral_perimeter_sum (c d : ℤ) :
  perimeter = c * Real.sqrt 5 + d * Real.sqrt 13 → c + d = 9 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l1566_156603


namespace NUMINAMATH_CALUDE_min_flowers_for_outstanding_pioneer_l1566_156608

/-- Represents the number of small red flowers needed for one small red flag -/
def flowers_per_flag : ℕ := 5

/-- Represents the number of small red flags needed for one badge -/
def flags_per_badge : ℕ := 4

/-- Represents the number of badges needed for one small gold cup -/
def badges_per_cup : ℕ := 3

/-- Represents the number of small gold cups needed to be an outstanding Young Pioneer -/
def cups_needed : ℕ := 2

/-- Theorem stating the minimum number of small red flowers needed to be an outstanding Young Pioneer -/
theorem min_flowers_for_outstanding_pioneer : 
  cups_needed * badges_per_cup * flags_per_badge * flowers_per_flag = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_flowers_for_outstanding_pioneer_l1566_156608


namespace NUMINAMATH_CALUDE_multiples_between_200_and_500_l1566_156610

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_200_and_500 : count_multiples 200 500 36 = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiples_between_200_and_500_l1566_156610


namespace NUMINAMATH_CALUDE_boxes_sold_tuesday_l1566_156640

/-- The number of boxes Kim sold on different days of the week -/
structure BoxesSold where
  friday : ℕ
  thursday : ℕ
  wednesday : ℕ
  tuesday : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : BoxesSold) : Prop :=
  b.friday = 600 ∧
  b.thursday = (3/2 : ℚ) * b.friday ∧
  b.wednesday = 2 * b.thursday ∧
  b.tuesday = 3 * b.wednesday

/-- The theorem stating that under the given conditions, Kim sold 5400 boxes on Tuesday -/
theorem boxes_sold_tuesday (b : BoxesSold) (h : problem_conditions b) : b.tuesday = 5400 := by
  sorry

end NUMINAMATH_CALUDE_boxes_sold_tuesday_l1566_156640


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1566_156631

theorem binomial_coefficient_ratio (a b : ℕ) : 
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k ≤ a) →
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k * 2^k ≤ b) →
  b / a = 12 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1566_156631


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l1566_156665

def num_islands : ℕ := 6
def num_treasure_islands : ℕ := 3

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  5/54 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l1566_156665


namespace NUMINAMATH_CALUDE_friend_of_gcd_l1566_156691

/-- Two integers are friends if their product is a perfect square -/
def are_friends (a b : ℤ) : Prop := ∃ k : ℤ, a * b = k * k

/-- Main theorem: If a is a friend of b, then a is a friend of gcd(a, b) -/
theorem friend_of_gcd {a b : ℤ} (h : are_friends a b) : are_friends a (Int.gcd a b) := by
  sorry

end NUMINAMATH_CALUDE_friend_of_gcd_l1566_156691


namespace NUMINAMATH_CALUDE_plane_equation_l1566_156611

/-- A plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def parallel (p1 p2 : Plane) : Prop :=
  p1.a * p2.b = p1.b * p2.a ∧ p1.a * p2.c = p1.c * p2.a ∧ p1.b * p2.c = p1.c * p2.b

def passes_through (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem plane_equation : 
  ∃ (plane : Plane), 
    plane.a = 3 ∧ 
    plane.b = -4 ∧ 
    plane.c = 1 ∧ 
    plane.d = 7 ∧ 
    passes_through plane ⟨2, 3, -1⟩ ∧ 
    parallel plane ⟨3, -4, 1, -5, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l1566_156611


namespace NUMINAMATH_CALUDE_fill_three_positions_from_fifteen_l1566_156654

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n < k then 0
  else n * fill_positions (n - 1) (k - 1)

/-- Theorem: There are 2730 ways to fill 3 positions from 15 candidates -/
theorem fill_three_positions_from_fifteen :
  fill_positions 15 3 = 2730 := by
  sorry

end NUMINAMATH_CALUDE_fill_three_positions_from_fifteen_l1566_156654


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1566_156615

/-- The equation of the tangent line to y = x³ + x + 1 at (1, 3) is 4x - y - 1 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + x + 1
  let point : ℝ × ℝ := (1, 3)
  let tangent_line (x y : ℝ) := 4*x - y - 1 = 0
  (∀ x, tangent_line x (f x)) ∧ tangent_line point.1 point.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l1566_156615


namespace NUMINAMATH_CALUDE_positive_difference_of_numbers_l1566_156689

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (square_diff_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_of_numbers_l1566_156689


namespace NUMINAMATH_CALUDE_tile_arrangements_example_l1566_156660

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 2 purple, 3 green, and 2 yellow tiles is 3780 -/
theorem tile_arrangements_example :
  tileArrangements 2 2 3 2 = 3780 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_example_l1566_156660


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l1566_156646

theorem square_minus_product_plus_square : 7^2 - 5*6 + 6^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l1566_156646


namespace NUMINAMATH_CALUDE_product_of_five_primes_with_491_l1566_156675

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_abc_abc (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)

theorem product_of_five_primes_with_491 :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime 491 ∧
    is_abc_abc (p₁ * p₂ * p₃ * p₄ * 491) ∧
    p₁ * p₂ * p₃ * p₄ * 491 = 982982 :=
  sorry

end NUMINAMATH_CALUDE_product_of_five_primes_with_491_l1566_156675


namespace NUMINAMATH_CALUDE_chelsea_victory_bullseyes_l1566_156694

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  totalShots : Nat
  shotsCompleted : Nat
  chelseaLead : Nat
  chelseaMinScore : Nat
  bullseyeScore : Nat

/-- Calculates the minimum number of bullseyes Chelsea needs to secure victory -/
def minBullseyesForVictory (contest : ArcheryContest) : Nat :=
  sorry

/-- Theorem stating the minimum number of bullseyes Chelsea needs is 67 -/
theorem chelsea_victory_bullseyes (contest : ArcheryContest) 
  (h1 : contest.totalShots = 150)
  (h2 : contest.shotsCompleted = 75)
  (h3 : contest.chelseaLead = 70)
  (h4 : contest.chelseaMinScore = 2)
  (h5 : contest.bullseyeScore = 10) :
  minBullseyesForVictory contest = 67 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_victory_bullseyes_l1566_156694


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1566_156629

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  let half_sum := (α + β) / 2
  let half_gamma := γ / 2
  (Real.sin half_sum - Real.cos half_gamma = 0) ∧
  (Real.tan half_gamma + Real.tan half_sum - (1 / Real.tan half_sum + 1 / Real.tan half_gamma) = 0) ∧
  (Real.sin half_sum ^ 2 + (1 / Real.tan half_sum) * (1 / Real.tan half_gamma) - Real.cos half_gamma ^ 2 = 1) ∧
  (Real.cos half_sum ^ 2 + Real.tan half_sum * Real.tan half_gamma + Real.cos half_gamma ^ 2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1566_156629


namespace NUMINAMATH_CALUDE_percentage_problem_l1566_156622

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 16) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1566_156622


namespace NUMINAMATH_CALUDE_cone_base_radius_l1566_156668

/-- Given a cone formed from a sector of a circle with a central angle of 120° and a radius of 6,
    the radius of the base circle of the cone is 2. -/
theorem cone_base_radius (sector_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  sector_angle = 120 * π / 180 ∧ 
  sector_radius = 6 ∧ 
  base_radius = sector_angle / (2 * π) * sector_radius → 
  base_radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1566_156668


namespace NUMINAMATH_CALUDE_problem_solution_l1566_156656

theorem problem_solution :
  (∀ x : ℝ, -3 * x * (2 * x^2 - x + 4) = -6 * x^3 + 3 * x^2 - 12 * x) ∧
  (∀ a b : ℝ, (2 * a - b) * (2 * a + b) = 4 * a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1566_156656


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1566_156604

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_prod : a 7 * a 9 = 4)
    (h_a4 : a 4 = 1) :
    a 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1566_156604


namespace NUMINAMATH_CALUDE_amount_r_holds_l1566_156612

theorem amount_r_holds (total : ℝ) (r_fraction : ℝ) (r_amount : ℝ) : 
  total = 7000 →
  r_fraction = 2/3 →
  r_amount = r_fraction * (total / (1 + r_fraction)) →
  r_amount = 2800 := by
sorry

end NUMINAMATH_CALUDE_amount_r_holds_l1566_156612


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1566_156666

theorem triangle_side_ratio (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are side lengths opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sine rule
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A + Real.cos A - 2 / (Real.sin B + Real.cos B) = 0 →
  -- Conclusion
  (a + b) / c = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1566_156666


namespace NUMINAMATH_CALUDE_gcf_of_450_and_210_l1566_156633

theorem gcf_of_450_and_210 : Nat.gcd 450 210 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_450_and_210_l1566_156633


namespace NUMINAMATH_CALUDE_ralphs_peanuts_l1566_156624

/-- Given Ralph's initial peanuts, lost peanuts, number of bags bought, and peanuts per bag,
    prove that Ralph ends up with the correct number of peanuts. -/
theorem ralphs_peanuts (initial : ℕ) (lost : ℕ) (bags : ℕ) (per_bag : ℕ)
    (h1 : initial = 2650)
    (h2 : lost = 1379)
    (h3 : bags = 4)
    (h4 : per_bag = 450) :
    initial - lost + bags * per_bag = 3071 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_peanuts_l1566_156624


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_l1566_156637

theorem grocer_coffee_stock (initial_stock : ℝ) : 
  initial_stock > 0 →
  0.30 * initial_stock + 0.60 * 100 = 0.36 * (initial_stock + 100) →
  initial_stock = 400 := by
sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_l1566_156637


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l1566_156673

/-- Calculates the required carpet area in square yards for a rectangular bedroom and square closet, including wastage. -/
theorem carpet_area_calculation 
  (bedroom_length : ℝ) 
  (bedroom_width : ℝ) 
  (closet_side : ℝ) 
  (wastage_rate : ℝ) 
  (feet_per_yard : ℝ) 
  (h1 : bedroom_length = 15)
  (h2 : bedroom_width = 10)
  (h3 : closet_side = 6)
  (h4 : wastage_rate = 0.1)
  (h5 : feet_per_yard = 3) :
  let bedroom_area := (bedroom_length / feet_per_yard) * (bedroom_width / feet_per_yard)
  let closet_area := (closet_side / feet_per_yard) ^ 2
  let total_area := bedroom_area + closet_area
  let required_area := total_area * (1 + wastage_rate)
  required_area = 22.715 := by
  sorry


end NUMINAMATH_CALUDE_carpet_area_calculation_l1566_156673


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_1202102_base5_l1566_156682

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_1202102_base5 :
  largestPrimeDivisor (base5ToBase10 1202102) = 307 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_1202102_base5_l1566_156682


namespace NUMINAMATH_CALUDE_cosine_symmetry_axis_l1566_156639

/-- Given a function f(x) = cos(x - π/4), prove that its axis of symmetry is x = π/4 + kπ where k ∈ ℤ -/
theorem cosine_symmetry_axis (f : ℝ → ℝ) (k : ℤ) :
  (∀ x, f x = Real.cos (x - π/4)) →
  (∀ x, f (π/4 + k * π + x) = f (π/4 + k * π - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_axis_l1566_156639


namespace NUMINAMATH_CALUDE_game_points_total_l1566_156636

theorem game_points_total (layla_points nahima_points : ℕ) : 
  layla_points = 70 → 
  layla_points = nahima_points + 28 → 
  layla_points + nahima_points = 112 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l1566_156636


namespace NUMINAMATH_CALUDE_billy_soda_distribution_l1566_156671

/-- Represents the number of sodas Billy can give to each sibling -/
def sodas_per_sibling (total_sodas : ℕ) (num_sisters : ℕ) : ℕ :=
  total_sodas / (num_sisters + 2 * num_sisters)

/-- Theorem stating that Billy can give 2 sodas to each sibling -/
theorem billy_soda_distribution :
  sodas_per_sibling 12 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_soda_distribution_l1566_156671


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1566_156609

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x ≤ 2}

-- Define set N
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1566_156609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1566_156690

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying
    the given conditions, the general term is 2n - 3. -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_mean1 : (a 2 + a 6) / 2 = 5)
    (h_mean2 : (a 3 + a 7) / 2 = 7) :
    ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1566_156690


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1566_156658

/-- The partial fraction decomposition equation holds for the given values of A, B, and C -/
theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x ≠ 1 → x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = A / x + (B * x + C) / (x^2 - 1) ∧
    A = 7 ∧ B = -9 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1566_156658


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1566_156613

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  x^(3/2) + 6*x^(5/4) + 8*x^(3/4) ≥ 15*x ∧
  (x^(3/2) + 6*x^(5/4) + 8*x^(3/4) = 15*x ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1566_156613


namespace NUMINAMATH_CALUDE_book_selling_price_l1566_156679

/-- Given a book with cost price CP, prove that the original selling price is 720 Rs. -/
theorem book_selling_price (CP : ℝ) : 
  (1.1 * CP = 880) →  -- Condition for 10% gain
  (∃ OSP, OSP = 0.9 * CP) →  -- Condition for 10% loss
  (∃ OSP, OSP = 720) :=
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l1566_156679


namespace NUMINAMATH_CALUDE_non_equilateral_triangle_coverage_l1566_156644

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- Define coverage of a triangle by two other triangles
def covers (t1 t2 t : Triangle) : Prop :=
  sorry

-- Define non-equilateral triangle
def nonEquilateral (t : Triangle) : Prop :=
  sorry

-- Define smaller triangle
def smaller (t1 t2 : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem non_equilateral_triangle_coverage (t : Triangle) :
  nonEquilateral t →
  ∃ (t1 t2 : Triangle), smaller t1 t ∧ smaller t2 t ∧ similar t1 t ∧ similar t2 t ∧ covers t1 t2 t :=
sorry

end NUMINAMATH_CALUDE_non_equilateral_triangle_coverage_l1566_156644


namespace NUMINAMATH_CALUDE_hcd_problem_l1566_156662

theorem hcd_problem : (Nat.gcd 2548 364 + 8) - 12 = 360 := by sorry

end NUMINAMATH_CALUDE_hcd_problem_l1566_156662


namespace NUMINAMATH_CALUDE_max_students_distribution_l1566_156614

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 3540) (h2 : pencils = 2860) :
  Nat.gcd pens pencils = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1566_156614


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1566_156650

theorem range_of_a_for_inequality : 
  {a : ℝ | ∃ x : ℝ, |x + 2| + |x - a| < 5} = Set.Ioo (-7 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1566_156650


namespace NUMINAMATH_CALUDE_correct_land_equation_l1566_156676

/-- Represents the relationship between arable land and forest land areas -/
def land_relationship (x y : ℝ) : Prop :=
  x + y = 2000 ∧ y = x * (30 / 100)

/-- The correct system of equations for the land areas -/
theorem correct_land_equation :
  ∀ x y : ℝ,
  (x + y = 2000 ∧ y = x * (30 / 100)) ↔ land_relationship x y :=
by sorry

end NUMINAMATH_CALUDE_correct_land_equation_l1566_156676


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l1566_156695

/-- The number of distinct terms in the expansion of (x+y)(a+b+c)(d+e+f) -/
def num_distinct_terms : ℕ := 18

/-- The first factor has 2 terms -/
def num_terms_factor1 : ℕ := 2

/-- The second factor has 3 terms -/
def num_terms_factor2 : ℕ := 3

/-- The third factor has 3 terms -/
def num_terms_factor3 : ℕ := 3

theorem expansion_distinct_terms :
  num_distinct_terms = num_terms_factor1 * num_terms_factor2 * num_terms_factor3 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l1566_156695


namespace NUMINAMATH_CALUDE_max_value_problem_1_l1566_156651

theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (max_y : ℝ), ∀ y : ℝ, y = 1/2 * x * (1 - 2*x) → y ≤ max_y ∧ max_y = 1/16 := by
  sorry


end NUMINAMATH_CALUDE_max_value_problem_1_l1566_156651


namespace NUMINAMATH_CALUDE_mac_loses_three_dollars_l1566_156645

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin * count : ℚ) / 100

/-- Represents Mac's trade of dimes for quarters -/
def dimes_for_quarter : ℕ := 3

/-- Represents Mac's trade of nickels for quarters -/
def nickels_for_quarter : ℕ := 7

/-- Number of quarters Mac trades for using dimes -/
def quarters_from_dimes : ℕ := 20

/-- Number of quarters Mac trades for using nickels -/
def quarters_from_nickels : ℕ := 20

/-- Theorem stating that Mac loses $3.00 in his trades -/
theorem mac_loses_three_dollars :
  total_value "quarter" (quarters_from_dimes + quarters_from_nickels) -
  (total_value "dime" (dimes_for_quarter * quarters_from_dimes) +
   total_value "nickel" (nickels_for_quarter * quarters_from_nickels)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_mac_loses_three_dollars_l1566_156645


namespace NUMINAMATH_CALUDE_triangle_division_perimeter_l1566_156667

/-- A structure representing a triangle division scenario -/
structure TriangleDivision where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The theorem statement -/
theorem triangle_division_perimeter 
  (td : TriangleDivision) 
  (h1 : td.large_perimeter = 120)
  (h2 : td.num_small_triangles = 9)
  (h3 : td.small_perimeter * 3 = td.large_perimeter) :
  td.small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_perimeter_l1566_156667


namespace NUMINAMATH_CALUDE_tv_show_total_cost_l1566_156627

def tv_show_cost (first_season_cost : ℕ) (first_season_episodes : ℕ) 
  (other_seasons_episodes : ℕ) (last_season_episodes : ℕ) (total_seasons : ℕ) : ℕ :=
  let other_seasons_cost := 2 * first_season_cost
  let first_season_total := first_season_cost * first_season_episodes
  let other_seasons_total := other_seasons_cost * 
    (other_seasons_episodes * (total_seasons - 2) + last_season_episodes)
  first_season_total + other_seasons_total

theorem tv_show_total_cost :
  tv_show_cost 100000 12 18 24 5 = 16800000 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_total_cost_l1566_156627


namespace NUMINAMATH_CALUDE_pm25_scientific_notation_l1566_156652

theorem pm25_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = -6 ∧ a = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pm25_scientific_notation_l1566_156652


namespace NUMINAMATH_CALUDE_function_properties_l1566_156623

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- State the theorem
theorem function_properties (a : ℝ) 
  (h1 : ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m ∧ ∃ (y : ℝ), f a y = m) 
  (h2 : ∀ (x : ℝ), f a x ≥ a) :
  (a = 2) ∧ 
  (∀ (x : ℝ), f 2 x ≤ 5 ↔ 1/2 ≤ x ∧ x ≤ 11/2) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1566_156623


namespace NUMINAMATH_CALUDE_stating_angle_bisector_division_l1566_156619

/-- Represents a parallelogram with sides of length 8 and 3 -/
structure Parallelogram where
  long_side : ℝ
  short_side : ℝ
  long_side_eq : long_side = 8
  short_side_eq : short_side = 3

/-- Represents the three parts of the divided side -/
structure DividedSide where
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ

/-- 
Theorem stating that the angle bisectors of the two angles adjacent to the longer side 
divide the opposite side into three parts with lengths 3, 2, and 3.
-/
theorem angle_bisector_division (p : Parallelogram) : 
  ∃ (d : DividedSide), d.part1 = 3 ∧ d.part2 = 2 ∧ d.part3 = 3 ∧ 
  d.part1 + d.part2 + d.part3 = p.long_side :=
sorry

end NUMINAMATH_CALUDE_stating_angle_bisector_division_l1566_156619


namespace NUMINAMATH_CALUDE_max_x_value_l1566_156621

theorem max_x_value (x y : ℝ) (h1 : x + y ≤ 1) (h2 : y + x + y ≤ 1) :
  x ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ ≤ 1 ∧ y₀ + x₀ + y₀ ≤ 1 ∧ x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l1566_156621


namespace NUMINAMATH_CALUDE_star_property_l1566_156664

/-- Custom binary operation ※ -/
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x - b * y

theorem star_property (a b : ℝ) (h : star a b 1 2 = 8) :
  star a b (-2) (-4) = -16 := by sorry

end NUMINAMATH_CALUDE_star_property_l1566_156664


namespace NUMINAMATH_CALUDE_last_date_2011_divisible_by_101_l1566_156686

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31 ∧
  (month ∈ [4, 6, 9, 11] → day ≤ 30) ∧
  (month = 2 → day ≤ 28)

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_2011_divisible_by_101 :
  ∀ year month day : ℕ,
    is_valid_date year month day →
    date_to_number year month day ≤ 20111221 →
    date_to_number year month day % 101 = 0 →
    date_to_number year month day = 20111221 :=
sorry

end NUMINAMATH_CALUDE_last_date_2011_divisible_by_101_l1566_156686


namespace NUMINAMATH_CALUDE_unsold_bars_l1566_156663

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_amount : ℕ) : 
  total_bars = 8 → price_per_bar = 4 → total_amount = 20 → 
  total_bars - (total_amount / price_per_bar) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unsold_bars_l1566_156663


namespace NUMINAMATH_CALUDE_no_square_on_four_circles_l1566_156698

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- Checks if four radii form a strictly increasing arithmetic progression -/
def is_strict_arithmetic_progression (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  ∃ (a d : ℝ), d > 0 ∧ r₁ = a ∧ r₂ = a + d ∧ r₃ = a + 2*d ∧ r₄ = a + 3*d

/-- Checks if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Main theorem statement -/
theorem no_square_on_four_circles (c₁ c₂ c₃ c₄ : Circle) 
  (h_common_center : c₁.center = c₂.center ∧ c₂.center = c₃.center ∧ c₃.center = c₄.center)
  (h_radii : is_strict_arithmetic_progression c₁.radius c₂.radius c₃.radius c₄.radius) :
  ¬ ∃ (s : Square), 
    (point_on_circle (s.vertices 0) c₁) ∧
    (point_on_circle (s.vertices 1) c₂) ∧
    (point_on_circle (s.vertices 2) c₃) ∧
    (point_on_circle (s.vertices 3) c₄) :=
by sorry

end NUMINAMATH_CALUDE_no_square_on_four_circles_l1566_156698


namespace NUMINAMATH_CALUDE_sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l1566_156648

theorem sufficient_condition_for_existence (m : ℝ) :
  (m ≤ 4) → (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) :=
by sorry

theorem necessary_condition_for_existence (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) → (m ≤ 4) :=
by sorry

theorem not_necessary_condition (m : ℝ) :
  ∃ m₀ : ℝ, m₀ ≤ 4 ∧ m₀ ≠ 4 ∧ (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m₀ ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l1566_156648


namespace NUMINAMATH_CALUDE_julian_facebook_friends_l1566_156601

theorem julian_facebook_friends :
  ∀ (julian_friends : ℕ) (julian_boys julian_girls boyd_boys boyd_girls : ℝ),
    julian_boys = 0.6 * julian_friends →
    julian_girls = 0.4 * julian_friends →
    boyd_girls = 2 * julian_girls →
    boyd_boys + boyd_girls = 100 →
    boyd_boys = 0.36 * 100 →
    julian_friends = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_julian_facebook_friends_l1566_156601


namespace NUMINAMATH_CALUDE_expression_simplification_l1566_156692

theorem expression_simplification (x : ℝ) : 
  3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1566_156692


namespace NUMINAMATH_CALUDE_pq_length_l1566_156684

/-- Two similar triangles PQR and STU with given side lengths and angles -/
structure SimilarTriangles where
  -- Side lengths of triangle PQR
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Side lengths of triangle STU
  ST : ℝ
  TU : ℝ
  SU : ℝ
  -- Angles
  angle_P : ℝ
  angle_S : ℝ
  -- Conditions
  h1 : angle_P = 120
  h2 : angle_S = 120
  h3 : PR = 15
  h4 : SU = 15
  h5 : ST = 4.5
  h6 : TU = 10.5

/-- The length of PQ in similar triangles PQR and STU is 9 -/
theorem pq_length (t : SimilarTriangles) : t.PQ = 9 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_l1566_156684


namespace NUMINAMATH_CALUDE_range_of_a_l1566_156618

/-- The equation x^2 + 2ax + 1 = 0 has two real roots greater than -1 -/
def p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁^2 + 2*a*x₁ + 1 = 0 ∧ x₂^2 + 2*a*x₂ + 1 = 0

/-- The solution set to the inequality ax^2 - ax + 1 > 0 is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a :
  (∀ a : ℝ, p a ∨ q a) →
  (∀ a : ℝ, ¬q a) →
  {a : ℝ | a ≤ -1} = {a : ℝ | p a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1566_156618


namespace NUMINAMATH_CALUDE_binomial_coefficient_six_choose_two_l1566_156672

theorem binomial_coefficient_six_choose_two : 
  Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_six_choose_two_l1566_156672


namespace NUMINAMATH_CALUDE_merchant_profit_theorem_l1566_156607

/-- Calculate the profit for a single item -/
def calculate_profit (purchase_price markup_percent discount_percent : ℚ) : ℚ :=
  let selling_price := purchase_price * (1 + markup_percent / 100)
  let discounted_price := selling_price * (1 - discount_percent / 100)
  discounted_price - purchase_price

/-- Calculate the total gross profit for three items -/
def total_gross_profit (
  jacket_price jeans_price shirt_price : ℚ)
  (jacket_markup jeans_markup shirt_markup : ℚ)
  (jacket_discount jeans_discount shirt_discount : ℚ) : ℚ :=
  calculate_profit jacket_price jacket_markup jacket_discount +
  calculate_profit jeans_price jeans_markup jeans_discount +
  calculate_profit shirt_price shirt_markup shirt_discount

theorem merchant_profit_theorem :
  total_gross_profit 60 45 30 25 30 15 20 10 5 = 10.43 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_theorem_l1566_156607


namespace NUMINAMATH_CALUDE_bound_on_c_l1566_156699

theorem bound_on_c (a b c : ℝ) 
  (sum_condition : a + 2 * b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  -2/3 ≤ c ∧ c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bound_on_c_l1566_156699


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1566_156606

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 50) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1125 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1566_156606


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1566_156655

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    (p ∣ (factorial 13 + factorial 14)) ∧
    (∀ q : ℕ, is_prime q → q ∣ (factorial 13 + factorial 14) → q ≤ p) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1566_156655


namespace NUMINAMATH_CALUDE_number_ordering_l1566_156687

theorem number_ordering : 
  let a := Real.log 0.32
  let b := Real.log 0.33
  let c := 20.3
  let d := 0.32
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_number_ordering_l1566_156687


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1566_156630

theorem sqrt_inequality : Real.sqrt 10 - Real.sqrt 5 < Real.sqrt 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1566_156630


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1566_156638

theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, 
    y = k * x + 2 * k + 1 ∧ 
    y = -1/2 * x + 2 ∧ 
    x > 0 ∧ 
    y > 0) ↔ 
  -1/6 < k ∧ k < 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1566_156638
