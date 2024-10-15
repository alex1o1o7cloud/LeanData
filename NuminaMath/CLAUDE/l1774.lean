import Mathlib

namespace NUMINAMATH_CALUDE_birdhouse_charge_for_two_l1774_177482

/-- The cost of a birdhouse for Denver -/
def birdhouse_cost (wood_pieces : ℕ) (wood_price paint_cost labor_cost : ℚ) : ℚ :=
  wood_pieces * wood_price + paint_cost + labor_cost

/-- The selling price of a birdhouse -/
def birdhouse_price (cost profit : ℚ) : ℚ :=
  cost + profit

/-- The total charge for multiple birdhouses -/
def total_charge (price : ℚ) (quantity : ℕ) : ℚ :=
  price * quantity

theorem birdhouse_charge_for_two :
  let wood_pieces : ℕ := 7
  let wood_price : ℚ := 3/2  -- $1.50
  let paint_cost : ℚ := 3
  let labor_cost : ℚ := 9/2  -- $4.50
  let profit : ℚ := 11/2  -- $5.50
  let cost := birdhouse_cost wood_pieces wood_price paint_cost labor_cost
  let price := birdhouse_price cost profit
  let quantity : ℕ := 2
  total_charge price quantity = 47
  := by sorry

end NUMINAMATH_CALUDE_birdhouse_charge_for_two_l1774_177482


namespace NUMINAMATH_CALUDE_drug_price_reduction_l1774_177475

/-- Proves that given an initial price of 100 yuan and a final price of 81 yuan
    after two equal percentage reductions, the average percentage reduction each time is 10% -/
theorem drug_price_reduction (initial_price : ℝ) (final_price : ℝ) (reduction_percentage : ℝ) :
  initial_price = 100 →
  final_price = 81 →
  final_price = initial_price * (1 - reduction_percentage)^2 →
  reduction_percentage = 0.1 := by
sorry


end NUMINAMATH_CALUDE_drug_price_reduction_l1774_177475


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1774_177499

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) = 60 * x^2 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1774_177499


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1774_177459

/-- A line parallel to 2x - y + 1 = 0 and tangent to x^2 + y^2 = 5 has equation 2x - y ± 5 = 0 -/
theorem tangent_line_equation (x y : ℝ) :
  ∃ (k : ℝ), k = 5 ∨ k = -5 ∧
  (∀ (x y : ℝ), 2*x - y + k = 0 →
    (∀ (x₀ y₀ : ℝ), 2*x₀ - y₀ + 1 = 0 → ∃ (t : ℝ), x = x₀ + 2*t ∧ y = y₀ + t) ∧
    (∃! (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 5 ∧ 2*x₀ - y₀ + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1774_177459


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1774_177457

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 + Real.sqrt 2
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1774_177457


namespace NUMINAMATH_CALUDE_remainder_2023_times_7_div_45_l1774_177431

theorem remainder_2023_times_7_div_45 : (2023 * 7) % 45 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2023_times_7_div_45_l1774_177431


namespace NUMINAMATH_CALUDE_clock_partition_exists_l1774_177436

/-- A partition of the set {1, 2, ..., 12} into three subsets -/
structure ClockPartition where
  part1 : Finset Nat
  part2 : Finset Nat
  part3 : Finset Nat
  partition_complete : part1 ∪ part2 ∪ part3 = Finset.range 12
  partition_disjoint1 : Disjoint part1 part2
  partition_disjoint2 : Disjoint part1 part3
  partition_disjoint3 : Disjoint part2 part3

/-- The theorem stating that there exists a partition of the clock numbers
    into three parts with equal sums -/
theorem clock_partition_exists : ∃ (p : ClockPartition),
  (p.part1.sum id = p.part2.sum id) ∧ (p.part2.sum id = p.part3.sum id) :=
sorry

end NUMINAMATH_CALUDE_clock_partition_exists_l1774_177436


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l1774_177480

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l1774_177480


namespace NUMINAMATH_CALUDE_gretchen_weekend_profit_l1774_177447

/-- Calculates Gretchen's profit from drawing caricatures over a weekend -/
def weekend_profit (
  full_body_price : ℕ)  -- Price of a full-body caricature
  (face_only_price : ℕ) -- Price of a face-only caricature
  (full_body_count : ℕ) -- Number of full-body caricatures drawn on Saturday
  (face_only_count : ℕ) -- Number of face-only caricatures drawn on Sunday
  (hourly_park_fee : ℕ) -- Hourly park fee
  (hours_per_day : ℕ)   -- Hours worked per day
  (art_supplies_cost : ℕ) -- Daily cost of art supplies
  : ℕ :=
  let total_revenue := full_body_price * full_body_count + face_only_price * face_only_count
  let total_park_fee := hourly_park_fee * hours_per_day * 2
  let total_supplies_cost := art_supplies_cost * 2
  let total_expenses := total_park_fee + total_supplies_cost
  total_revenue - total_expenses

/-- Theorem stating Gretchen's profit for the weekend -/
theorem gretchen_weekend_profit :
  weekend_profit 25 15 24 16 5 6 8 = 764 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_weekend_profit_l1774_177447


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_specific_prism_volume_l1774_177483

/-- A right rectangular prism with face areas a, b, and c has volume equal to the square root of their product. -/
theorem rectangular_prism_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = a ∧ w * h = b ∧ l * h = c ∧
  l * w * h = Real.sqrt (a * b * c) := by
  sorry

/-- The volume of a right rectangular prism with face areas 10, 14, and 35 square inches is 70 cubic inches. -/
theorem specific_prism_volume :
  ∃ l w h : ℝ, l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 10 ∧ w * h = 14 ∧ l * h = 35 ∧
  l * w * h = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_specific_prism_volume_l1774_177483


namespace NUMINAMATH_CALUDE_initial_candies_count_l1774_177487

/-- The number of candies sold on Day 1 -/
def day1_sales : ℕ := 1249

/-- The additional number of candies sold on Day 2 compared to Day 1 -/
def day2_additional : ℕ := 328

/-- The additional number of candies sold on Day 3 compared to Day 2 -/
def day3_additional : ℕ := 275

/-- The number of candies remaining after three days of sales -/
def remaining_candies : ℕ := 367

/-- The total number of candies at the beginning -/
def initial_candies : ℕ := day1_sales + (day1_sales + day2_additional) + (day1_sales + day2_additional + day3_additional) + remaining_candies

theorem initial_candies_count : initial_candies = 5045 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_count_l1774_177487


namespace NUMINAMATH_CALUDE_prob_6_to_7_l1774_177438

-- Define a normally distributed random variable
def X : Real → Real := sorry

-- Define the probability density function for X
def pdf (x : Real) : Real := sorry

-- Define the cumulative distribution function for X
def cdf (x : Real) : Real := sorry

-- Given probabilities
axiom prob_1sigma : (cdf 6 - cdf 4) = 0.6826
axiom prob_2sigma : (cdf 7 - cdf 3) = 0.9544
axiom prob_3sigma : (cdf 8 - cdf 2) = 0.9974

-- The statement to prove
theorem prob_6_to_7 : (cdf 7 - cdf 6) = 0.1359 := by sorry

end NUMINAMATH_CALUDE_prob_6_to_7_l1774_177438


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l1774_177413

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if a point is inside a triangle -/
def isInside (p : Point3D) (t : Tetrahedron) : Prop :=
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = Point3D.mk (a * t.A.x + b * t.B.x + c * t.C.x)
                   (a * t.A.y + b * t.B.y + c * t.C.y)
                   (a * t.A.z + b * t.B.z + c * t.C.z)

/-- Calculate the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Find the intersection point of a line parallel to DD₁ passing through a vertex -/
noncomputable def intersectionPoint (t : Tetrahedron) (D₁ : Point3D) (vertex : Point3D) : Point3D := sorry

/-- The main theorem -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) (D₁ : Point3D) :
  isInside D₁ t →
  let A₁ := intersectionPoint t D₁ t.A
  let B₁ := intersectionPoint t D₁ t.B
  let C₁ := intersectionPoint t D₁ t.C
  let t₁ := Tetrahedron.mk A₁ B₁ C₁ D₁
  volume t = (1/3) * volume t₁ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l1774_177413


namespace NUMINAMATH_CALUDE_range_of_a_for_zero_point_solution_for_specific_a_l1774_177455

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

-- Theorem for the range of a
theorem range_of_a_for_zero_point :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ f a x = 0) ↔
    a ∈ Set.Icc (12 * (27 - 4 * Real.sqrt 6) / 211) (12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

-- Theorem for the specific solution when a = 32/17
theorem solution_for_specific_a :
  ∃ x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ f (32/17) x = 0 ∧ x = 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_zero_point_solution_for_specific_a_l1774_177455


namespace NUMINAMATH_CALUDE_circle_area_difference_l1774_177452

theorem circle_area_difference : 
  let r1 : ℝ := 15
  let d2 : ℝ := 14
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 176 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1774_177452


namespace NUMINAMATH_CALUDE_initial_persimmons_l1774_177444

/-- The number of persimmons eaten -/
def eaten : ℕ := 5

/-- The number of persimmons left -/
def left : ℕ := 12

/-- The initial number of persimmons -/
def initial : ℕ := eaten + left

theorem initial_persimmons : initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_persimmons_l1774_177444


namespace NUMINAMATH_CALUDE_number_of_cars_l1774_177498

theorem number_of_cars (total_distance : ℝ) (car_spacing : ℝ) (h1 : total_distance = 242) (h2 : car_spacing = 5.5) :
  ⌊total_distance / car_spacing⌋ + 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cars_l1774_177498


namespace NUMINAMATH_CALUDE_eggs_in_boxes_l1774_177464

theorem eggs_in_boxes (eggs_per_box : ℕ) (num_boxes : ℕ) :
  eggs_per_box = 15 → num_boxes = 7 → eggs_per_box * num_boxes = 105 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_boxes_l1774_177464


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l1774_177451

/-- The percentage of filled seats in a public show -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem stating that the percentage of filled seats is 62% -/
theorem seats_filled_percentage (total_seats vacant_seats : ℕ)
  (h1 : total_seats = 600)
  (h2 : vacant_seats = 228) :
  percentage_filled total_seats vacant_seats = 62 := by
  sorry

end NUMINAMATH_CALUDE_seats_filled_percentage_l1774_177451


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1774_177405

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perp_line (n l : Line) : Prop := sorry

/-- Main theorem -/
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m n : Line) 
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at α β l)
  (h3 : line_parallel_plane m α)
  (h4 : line_perp_plane n β) :
  line_perp_line n l := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1774_177405


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1774_177450

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1774_177450


namespace NUMINAMATH_CALUDE_unique_solution_for_b_l1774_177407

def base_75_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (75 ^ i)) 0

theorem unique_solution_for_b : ∃! b : ℕ, 
  0 ≤ b ∧ b ≤ 19 ∧ 
  (base_75_to_decimal [9, 2, 4, 6, 1, 8, 7, 2, 5] - b) % 17 = 0 ∧
  b = 8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_b_l1774_177407


namespace NUMINAMATH_CALUDE_star_calculation_l1774_177414

/-- The custom operation ⋆ defined as x ⋆ y = (x² + y²)(x - y) -/
def star (x y : ℝ) : ℝ := (x^2 + y^2) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = 16983 -/
theorem star_calculation : star 2 (star 3 4) = 16983 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1774_177414


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_129_l1774_177402

theorem first_nonzero_digit_of_one_over_129 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 129 = (n : ℚ) / 10^(n+1) + r ∧ 0 ≤ r ∧ r < 1 / 10^(n+1) ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_129_l1774_177402


namespace NUMINAMATH_CALUDE_factorial_ratio_l1774_177476

theorem factorial_ratio : Nat.factorial 15 / Nat.factorial 14 = 15 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1774_177476


namespace NUMINAMATH_CALUDE_f_composition_of_i_l1774_177421

noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z^2 + 1 else -z^2 - 1

theorem f_composition_of_i : f (f (f (f Complex.I))) = -26 := by sorry

end NUMINAMATH_CALUDE_f_composition_of_i_l1774_177421


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l1774_177411

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 8)^2 = 5 ↔ x^2 - 16*x + 59 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l1774_177411


namespace NUMINAMATH_CALUDE_bonus_difference_l1774_177469

/-- Prove that given a total bonus of $5,000 divided between two employees,
    where the senior employee receives $1,900 and the junior employee receives $3,100,
    the difference between the junior employee's bonus and the senior employee's bonus is $1,200. -/
theorem bonus_difference (total_bonus senior_bonus junior_bonus : ℕ) : 
  total_bonus = 5000 →
  senior_bonus = 1900 →
  junior_bonus = 3100 →
  junior_bonus - senior_bonus = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bonus_difference_l1774_177469


namespace NUMINAMATH_CALUDE_angle_side_relationship_l1774_177472

-- Define a triangle with angles A, B, C and sides a, b, c
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Triangle inequality
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  -- Angle sum in a triangle is π
  angle_sum : A + B + C = π
  -- Side lengths satisfy triangle inequality
  side_ineq_a : a < b + c
  side_ineq_b : b < a + c
  side_ineq_c : c < a + b

theorem angle_side_relationship (t : Triangle) : t.A > t.B ↔ t.a > t.b := by
  sorry

end NUMINAMATH_CALUDE_angle_side_relationship_l1774_177472


namespace NUMINAMATH_CALUDE_clock_tower_rings_per_year_l1774_177412

/-- The number of times a clock tower bell rings in a year -/
def bell_rings_per_year (rings_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : ℕ :=
  rings_per_hour * hours_per_day * days_per_year

/-- Theorem: The clock tower bell rings 8760 times in a year -/
theorem clock_tower_rings_per_year :
  bell_rings_per_year 1 24 365 = 8760 := by
  sorry

end NUMINAMATH_CALUDE_clock_tower_rings_per_year_l1774_177412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1774_177425

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1774_177425


namespace NUMINAMATH_CALUDE_non_zero_digits_count_l1774_177493

/-- The fraction we're working with -/
def f : ℚ := 120 / (2^5 * 5^9)

/-- Count of non-zero digits after the decimal point in the decimal representation of a rational number -/
noncomputable def count_non_zero_digits_after_decimal (q : ℚ) : ℕ := sorry

/-- The main theorem: the count of non-zero digits after the decimal point for our fraction is 2 -/
theorem non_zero_digits_count : count_non_zero_digits_after_decimal f = 2 := by sorry

end NUMINAMATH_CALUDE_non_zero_digits_count_l1774_177493


namespace NUMINAMATH_CALUDE_work_completion_time_l1774_177448

/-- The time it takes for A, B, and C to complete a work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work in 2 days -/
theorem work_completion_time :
  time_together 4 10 (20 / 3) = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1774_177448


namespace NUMINAMATH_CALUDE_min_constant_for_sqrt_inequality_l1774_177424

theorem min_constant_for_sqrt_inequality :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_min_constant_for_sqrt_inequality_l1774_177424


namespace NUMINAMATH_CALUDE_wire_service_reporters_l1774_177453

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (politics : ℝ) : 
  local_politics = 0.2 * total →
  local_politics = 0.8 * politics →
  (total - politics) / total = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l1774_177453


namespace NUMINAMATH_CALUDE_peach_pie_slices_count_l1774_177471

/-- Represents the number of slices in a peach pie -/
def peach_pie_slices : ℕ := sorry

/-- Represents the number of slices in an apple pie -/
def apple_pie_slices : ℕ := 8

/-- Represents the number of customers who ordered apple pie slices -/
def apple_pie_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_pie_customers : ℕ := 48

/-- Represents the total number of pies sold during the weekend -/
def total_pies_sold : ℕ := 15

theorem peach_pie_slices_count : peach_pie_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_pie_slices_count_l1774_177471


namespace NUMINAMATH_CALUDE_tangent_and_mean_value_theorem_l1774_177462

noncomputable section

/-- The function f(x) = x^2 + a(x + ln x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * (x + Real.log x)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + a * (1 + 1/x)

theorem tangent_and_mean_value_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f_deriv a x₀ = 2*(a+1) ∧ f a x₀ = (a+1)*(2*x₀-1) - a - 1) ∧
  (∃ ξ : ℝ, 1 < ξ ∧ ξ < Real.exp 1 ∧ f_deriv a ξ = (f a (Real.exp 1) - f a 1) / (Real.exp 1 - 1)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_and_mean_value_theorem_l1774_177462


namespace NUMINAMATH_CALUDE_inequality_proof_l1774_177409

theorem inequality_proof (x m : ℝ) (hx : x ≥ 1) (hm : m ≥ 1/2) :
  x * Real.log x ≤ m * (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1774_177409


namespace NUMINAMATH_CALUDE_grasshopper_position_l1774_177416

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Represents a single jump of the grasshopper -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.seven
  | Point.two => Point.seven
  | Point.three => Point.two
  | Point.four => Point.two
  | Point.five => Point.four
  | Point.six => Point.four
  | Point.seven => Point.six

/-- Represents multiple jumps of the grasshopper -/
def multi_jump (p : Point) (n : Nat) : Point :=
  match n with
  | 0 => p
  | Nat.succ m => jump (multi_jump p m)

theorem grasshopper_position : multi_jump Point.seven 2011 = Point.two := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_position_l1774_177416


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l1774_177440

theorem fixed_point_on_graph (m : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 9 * x^2 + m * x - 5 * m
  f 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l1774_177440


namespace NUMINAMATH_CALUDE_password_probability_l1774_177443

/-- Represents the composition of a password --/
structure Password :=
  (first_letter : Char)
  (middle_digit : Nat)
  (last_letter : Char)

/-- Defines the set of vowels --/
def vowels : Set Char := {'A', 'E', 'I', 'O', 'U'}

/-- Defines the set of even single-digit numbers --/
def even_single_digits : Set Nat := {0, 2, 4, 6, 8}

/-- The total number of letters in the alphabet --/
def alphabet_size : Nat := 26

/-- The number of vowels --/
def vowel_count : Nat := 5

/-- The number of single-digit numbers --/
def single_digit_count : Nat := 10

/-- The number of even single-digit numbers --/
def even_single_digit_count : Nat := 5

/-- Theorem stating the probability of a specific password pattern --/
theorem password_probability :
  (((vowel_count : ℚ) / alphabet_size) *
   ((even_single_digit_count : ℚ) / single_digit_count) *
   ((alphabet_size - vowel_count : ℚ) / alphabet_size)) =
  (105 : ℚ) / 1352 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l1774_177443


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1774_177489

/-- Given a quadratic function f(x) = x^2 + px + qx where p and q are positive constants,
    prove that the x-coordinate of its minimum value occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q*x
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1774_177489


namespace NUMINAMATH_CALUDE_exponent_sum_l1774_177418

theorem exponent_sum (a x y : ℝ) (hx : a^x = 2) (hy : a^y = 3) : a^(x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l1774_177418


namespace NUMINAMATH_CALUDE_largest_integer_difference_in_triangle_l1774_177497

theorem largest_integer_difference_in_triangle (n : ℕ) (hn : n ≥ 4) :
  (∃ k : ℕ, k > 0 ∧
    (∀ k' : ℕ, k' > k →
      ¬∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
        c - b ≥ k' ∧ b - a ≥ k' ∧ a + b ≥ c + 1) ∧
    (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
      c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1)) ∧
  (∀ k : ℕ, (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
    c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1) →
    k ≤ (n - 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_difference_in_triangle_l1774_177497


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l1774_177491

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l1774_177491


namespace NUMINAMATH_CALUDE_fred_initial_balloons_l1774_177400

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- The initial number of green balloons Fred had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem fred_initial_balloons : initial_balloons = 709 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_balloons_l1774_177400


namespace NUMINAMATH_CALUDE_johns_initial_speed_johns_initial_speed_proof_l1774_177417

theorem johns_initial_speed 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) : ℝ :=
  let final_time := initial_time * (1 + time_increase_percent / 100)
  let initial_speed := (final_distance / final_time) - speed_increase
  initial_speed

#check johns_initial_speed 8 75 4 168 = 8

theorem johns_initial_speed_proof 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) :
  johns_initial_speed initial_time time_increase_percent speed_increase final_distance = 8 :=
by sorry

end NUMINAMATH_CALUDE_johns_initial_speed_johns_initial_speed_proof_l1774_177417


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1774_177406

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 135 → 
  (n - 2) * 180 = n * interior_angle → 
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1774_177406


namespace NUMINAMATH_CALUDE_count_valid_permutations_l1774_177430

def alphabet : List Char := ['a', 'b', 'c', 'd', 'e']

def is_adjacent (c1 c2 : Char) : Bool :=
  let idx1 := alphabet.indexOf c1
  let idx2 := alphabet.indexOf c2
  (idx1 + 1 = idx2) || (idx2 + 1 = idx1)

def is_valid_permutation (perm : List Char) : Bool :=
  List.zip perm (List.tail perm) |>.all (fun (c1, c2) => !is_adjacent c1 c2)

def valid_permutations : List (List Char) :=
  List.permutations alphabet |>.filter is_valid_permutation

theorem count_valid_permutations : valid_permutations.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_permutations_l1774_177430


namespace NUMINAMATH_CALUDE_music_club_ratio_l1774_177434

theorem music_club_ratio :
  ∀ (total girls boys : ℕ) (p_girl p_boy : ℝ),
    total = girls + boys →
    total > 0 →
    p_girl + p_boy = 1 →
    p_girl = (3 / 5 : ℝ) * p_boy →
    (girls : ℝ) / total = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_music_club_ratio_l1774_177434


namespace NUMINAMATH_CALUDE_new_england_population_l1774_177456

/-- The population of New York -/
def population_NY : ℕ := sorry

/-- The population of New England -/
def population_NE : ℕ := sorry

/-- New York's population is two-thirds of New England's -/
axiom ny_two_thirds_ne : population_NY = (2 * population_NE) / 3

/-- The combined population of New York and New England is 3,500,000 -/
axiom combined_population : population_NY + population_NE = 3500000

/-- Theorem: The population of New England is 2,100,000 -/
theorem new_england_population : population_NE = 2100000 := by sorry

end NUMINAMATH_CALUDE_new_england_population_l1774_177456


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1774_177422

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1774_177422


namespace NUMINAMATH_CALUDE_direction_vector_proof_l1774_177439

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a vector is a direction vector of a line -/
def isDirectionVector (l : Line2D) (v : Vector2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- The given line 4x - 3y + m = 0 -/
def givenLine : Line2D :=
  { a := 4, b := -3, c := 0 }  -- We set c to 0 as 'm' is arbitrary

/-- The vector (3, 4) -/
def givenVector : Vector2D :=
  { x := 3, y := 4 }

/-- Theorem: (3, 4) is a direction vector of the line 4x - 3y + m = 0 -/
theorem direction_vector_proof : 
  isDirectionVector givenLine givenVector := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_proof_l1774_177439


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l1774_177490

-- Define the lines and point
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + m * y - 1 = 0
def line2 (n : ℝ) (x y : ℝ) : Prop := 3 * x - 2 * y + n = 0
def foot (p : ℝ) : ℝ × ℝ := (2, p)

-- State the theorem
theorem perpendicular_lines_sum (m n p : ℝ) : 
  (∀ x y, line1 m x y → line2 n x y → (x - 2) * (3 * x - 2 * y + n) + (y - p) * (2 * x + m * y - 1) = 0) →  -- perpendicularity condition
  line1 m 2 p →  -- foot satisfies line1
  line2 n 2 p →  -- foot satisfies line2
  m + n + p = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l1774_177490


namespace NUMINAMATH_CALUDE_divides_m_implies_divides_m_times_n_plus_one_l1774_177467

theorem divides_m_implies_divides_m_times_n_plus_one (m n : ℤ) :
  n ∣ m * (n + 1) → n ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divides_m_implies_divides_m_times_n_plus_one_l1774_177467


namespace NUMINAMATH_CALUDE_some_number_value_l1774_177403

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 7) :
  some_number = 105 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1774_177403


namespace NUMINAMATH_CALUDE_inequality_solution_l1774_177484

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -11/6 ∨ x > -4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1774_177484


namespace NUMINAMATH_CALUDE_simplify_fraction_l1774_177442

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1774_177442


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l1774_177420

theorem triangle_side_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom_seq : b^2 = a*c) :
  2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l1774_177420


namespace NUMINAMATH_CALUDE_discount_composition_l1774_177435

theorem discount_composition (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.4
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_discount := 1 - (price_after_second / original_price)
  total_discount = 0.58 := by
sorry

end NUMINAMATH_CALUDE_discount_composition_l1774_177435


namespace NUMINAMATH_CALUDE_trader_pens_sold_l1774_177441

/-- Calculates the number of pens sold given the gain and gain percentage -/
def pens_sold (gain_in_pens : ℕ) (gain_percentage : ℕ) : ℕ :=
  (gain_in_pens * 100) / gain_percentage

theorem trader_pens_sold : pens_sold 40 40 = 100 := by
  sorry

end NUMINAMATH_CALUDE_trader_pens_sold_l1774_177441


namespace NUMINAMATH_CALUDE_probability_multiple_2_3_7_l1774_177463

/-- The number of integers from 1 to n that are divisible by at least one of a, b, or c -/
def countMultiples (n : ℕ) (a b c : ℕ) : ℕ :=
  (n / a + n / b + n / c) - (n / lcm a b + n / lcm a c + n / lcm b c) + n / lcm a (lcm b c)

/-- The probability of selecting a multiple of 2, 3, or 7 from the first 150 positive integers -/
theorem probability_multiple_2_3_7 : 
  countMultiples 150 2 3 7 = 107 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_2_3_7_l1774_177463


namespace NUMINAMATH_CALUDE_smallest_multiple_of_11_23_37_l1774_177404

theorem smallest_multiple_of_11_23_37 : ∃ (n : ℕ), n > 0 ∧ 11 ∣ n ∧ 23 ∣ n ∧ 37 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 11 ∣ m ∧ 23 ∣ m ∧ 37 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_11_23_37_l1774_177404


namespace NUMINAMATH_CALUDE_problem_statements_l1774_177426

theorem problem_statements :
  (∀ (x : ℝ), x ≥ 3 → 2*x - 10 ≥ 0) ↔ ¬(∃ (x : ℝ), x ≥ 3 ∧ 2*x - 10 < 0) ∧
  (∀ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ (a b m : ℝ), a > b ∧ b > 0 ∧ m > 0 → a / b > (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1774_177426


namespace NUMINAMATH_CALUDE_amusement_park_admission_difference_l1774_177429

theorem amusement_park_admission_difference :
  let students : ℕ := 194
  let adults : ℕ := 235
  let free_admission : ℕ := 68
  let total_visitors : ℕ := students + adults
  let paid_admission : ℕ := total_visitors - free_admission
  paid_admission - free_admission = 293 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_difference_l1774_177429


namespace NUMINAMATH_CALUDE_skew_lines_distance_l1774_177454

/-- Given two skew lines a and b forming an angle θ, with their common perpendicular AA' of length d,
    and points E on a and F on b such that A'E = m and AF = n, the distance EF is given by
    √(d² + m² + n² ± 2mn cos θ). -/
theorem skew_lines_distance (θ d m n : ℝ) : ∃ (EF : ℝ), 
  EF = Real.sqrt (d^2 + m^2 + n^2 + 2*m*n*(Real.cos θ)) ∨
  EF = Real.sqrt (d^2 + m^2 + n^2 - 2*m*n*(Real.cos θ)) :=
by sorry


end NUMINAMATH_CALUDE_skew_lines_distance_l1774_177454


namespace NUMINAMATH_CALUDE_reunion_attendance_overlap_l1774_177427

theorem reunion_attendance_overlap (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ) (brown_attendees : ℕ)
  (h_total : total_guests = 200)
  (h_oates : oates_attendees = 60)
  (h_hall : hall_attendees = 90)
  (h_brown : brown_attendees = 80)
  (h_all_attend : total_guests ≤ oates_attendees + hall_attendees + brown_attendees) :
  let min_overlap := oates_attendees + hall_attendees + brown_attendees - total_guests
  let max_overlap := min oates_attendees (min hall_attendees brown_attendees)
  (min_overlap = 30 ∧ max_overlap = 60) :=
by sorry

end NUMINAMATH_CALUDE_reunion_attendance_overlap_l1774_177427


namespace NUMINAMATH_CALUDE_equation_solution_and_expression_value_l1774_177481

theorem equation_solution_and_expression_value :
  ∃ y : ℝ, (4 * y - 8 = 2 * y + 18) ∧ (3 * (y^2 + 6 * y + 12) = 777) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_expression_value_l1774_177481


namespace NUMINAMATH_CALUDE_product_eleven_sum_possibilities_l1774_177446

theorem product_eleven_sum_possibilities (a b c : ℤ) : 
  a * b * c = -11 → (a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_sum_possibilities_l1774_177446


namespace NUMINAMATH_CALUDE_three_cards_different_suits_l1774_177433

/-- The number of suits in a standard deck of cards -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 3

/-- The total number of ways to choose 3 cards from a standard deck of 52 cards,
    where all three cards are of different suits and the order doesn't matter -/
def ways_to_choose : ℕ := num_suits.choose cards_to_choose * cards_per_suit ^ cards_to_choose

theorem three_cards_different_suits :
  ways_to_choose = 8788 := by sorry

end NUMINAMATH_CALUDE_three_cards_different_suits_l1774_177433


namespace NUMINAMATH_CALUDE_computer_price_difference_l1774_177410

/-- The price difference between two stores selling the same computer with different prices and discounts -/
theorem computer_price_difference (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (discount2 : ℝ) 
  (h1 : price1 = 950) (h2 : discount1 = 0.06) (h3 : price2 = 920) (h4 : discount2 = 0.05) :
  abs (price1 * (1 - discount1) - price2 * (1 - discount2)) = 19 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_difference_l1774_177410


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1774_177401

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1774_177401


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_15_l1774_177437

theorem binomial_coefficient_22_15 (h1 : Nat.choose 21 13 = 20349)
                                   (h2 : Nat.choose 21 14 = 11628)
                                   (h3 : Nat.choose 23 15 = 490314) :
  Nat.choose 22 15 = 458337 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_15_l1774_177437


namespace NUMINAMATH_CALUDE_football_likers_l1774_177460

theorem football_likers (total : ℕ) (likers : ℕ) (players : ℕ) : 
  (24 : ℚ) / total = (likers : ℚ) / 250 →
  (players : ℚ) / likers = 1 / 2 →
  players = 50 →
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_football_likers_l1774_177460


namespace NUMINAMATH_CALUDE_binomial_expansion_cube_problem_solution_l1774_177445

theorem binomial_expansion_cube (x : ℕ) :
  x^3 + 3*(x^2) + 3*x + 1 = (x + 1)^3 :=
by sorry

theorem problem_solution : 
  85^3 + 3*(85^2) + 3*85 + 1 = 636256 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_cube_problem_solution_l1774_177445


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1774_177494

theorem intersection_x_coordinate (x y : ℤ) : 
  (y ≡ 3 * x + 4 [ZMOD 9]) → 
  (y ≡ 7 * x + 2 [ZMOD 9]) → 
  (x ≡ 5 [ZMOD 9]) := by
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1774_177494


namespace NUMINAMATH_CALUDE_product_of_roots_l1774_177496

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 8 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 8 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 8) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1774_177496


namespace NUMINAMATH_CALUDE_solar_panel_installation_l1774_177478

/-- The number of homes that can have solar panels fully installed given the total number of homes,
    panels required per home, and the shortage in supplied panels. -/
def homes_with_panels (total_homes : ℕ) (panels_per_home : ℕ) (panel_shortage : ℕ) : ℕ :=
  ((total_homes * panels_per_home - panel_shortage) / panels_per_home)

/-- Theorem stating that given 20 homes, each requiring 10 solar panels, and a supplier bringing
    50 panels less than required, the number of homes that can have their panels fully installed is 15. -/
theorem solar_panel_installation :
  homes_with_panels 20 10 50 = 15 := by
  sorry

#eval homes_with_panels 20 10 50

end NUMINAMATH_CALUDE_solar_panel_installation_l1774_177478


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l1774_177415

-- Define the pentagon and extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions
axiom ext_A : A' = A + (A - B)
axiom ext_B : B' = B + (B - C)
axiom ext_C : C' = C + (C - D)
axiom ext_D : D' = D + (D - E)
axiom ext_E : E' = E + (E - A)

-- Define the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (5/31 : ℝ) • B' + (10/31 : ℝ) • C' + (15/31 : ℝ) • D' + (1/31 : ℝ) • E' := by
  sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l1774_177415


namespace NUMINAMATH_CALUDE_harry_total_cost_l1774_177458

-- Define the conversion rate
def silver_per_gold : ℕ := 9

-- Define the costs
def spellbook_cost_gold : ℕ := 5
def potion_kit_cost_silver : ℕ := 20
def owl_cost_gold : ℕ := 28

-- Define the quantities
def num_spellbooks : ℕ := 5
def num_potion_kits : ℕ := 3
def num_owls : ℕ := 1

-- Define the total cost function
def total_cost_silver : ℕ :=
  (num_spellbooks * spellbook_cost_gold * silver_per_gold) +
  (num_potion_kits * potion_kit_cost_silver) +
  (num_owls * owl_cost_gold * silver_per_gold)

-- Theorem statement
theorem harry_total_cost :
  total_cost_silver = 537 :=
by sorry

end NUMINAMATH_CALUDE_harry_total_cost_l1774_177458


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1774_177449

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1774_177449


namespace NUMINAMATH_CALUDE_count_ten_digit_numbers_theorem_l1774_177466

/-- Count of ten-digit numbers with a given digit sum -/
def count_ten_digit_numbers (n : ℕ) : ℕ :=
  match n with
  | 2 => 46
  | 3 => 166
  | 4 => 361
  | _ => 0

/-- Theorem stating the count of ten-digit numbers with specific digit sums -/
theorem count_ten_digit_numbers_theorem :
  (count_ten_digit_numbers 2 = 46) ∧
  (count_ten_digit_numbers 3 = 166) ∧
  (count_ten_digit_numbers 4 = 361) := by
  sorry

end NUMINAMATH_CALUDE_count_ten_digit_numbers_theorem_l1774_177466


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1774_177468

/-- Given a large square with side length 8y and a smaller central square with side length 3y,
    where the large square is divided into the smaller central square and four congruent rectangles,
    the perimeter of one of these rectangles is 16y. -/
theorem rectangle_perimeter (y : ℝ) : 
  let large_square_side : ℝ := 8 * y
  let small_square_side : ℝ := 3 * y
  let rectangle_width : ℝ := small_square_side
  let rectangle_height : ℝ := large_square_side - small_square_side
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_height)
  rectangle_perimeter = 16 * y :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1774_177468


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1774_177423

/-- Given two adjacent points (1,2) and (2,5) on a square in a Cartesian coordinate plane,
    the area of the square is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (2, 5)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1774_177423


namespace NUMINAMATH_CALUDE_not_perfect_square_sum_of_squares_l1774_177492

theorem not_perfect_square_sum_of_squares (x y : ℤ) :
  ¬ ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 = n^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_sum_of_squares_l1774_177492


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l1774_177470

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

/-- The equation of a hyperbola -/
def is_hyperbola (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

theorem hyperbola_a_value (a : ℝ) :
  a > 0 →
  focal_length (Real.sqrt 10) = 2 * Real.sqrt 10 →
  is_hyperbola a (Real.sqrt 6) (Real.sqrt 10) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l1774_177470


namespace NUMINAMATH_CALUDE_unique_power_sum_l1774_177477

theorem unique_power_sum (k : ℕ) : (∃ (n t : ℕ), t ≥ 2 ∧ 3^k + 5^k = n^t) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_power_sum_l1774_177477


namespace NUMINAMATH_CALUDE_parabola_equation_l1774_177486

/-- A parabola with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (F : ℝ × ℝ) (h : F = (-2, 0)) : 
  ∃ (x y : ℝ), y^2 = -8*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1774_177486


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1774_177465

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1774_177465


namespace NUMINAMATH_CALUDE_incorrect_statement_l1774_177461

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the containment relation for lines and planes
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) : 
  ¬(∀ α β m n, parallelLinePlane m α ∧ intersect α β = n → parallelLine m n) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1774_177461


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1774_177473

theorem quadratic_polynomial_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_distinct_a_b : a ≠ b) (h_distinct_b_c : b ≠ c) (h_distinct_a_c : a ≠ c)
  (h_quadratic : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_f_a : f a = b * c) (h_f_b : f b = c * a) (h_f_c : f c = a * b) :
  f (a + b + c) = a * b + b * c + a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1774_177473


namespace NUMINAMATH_CALUDE_quadI_area_less_than_quadII_area_l1774_177432

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

/-- Quadrilateral I with vertices (0,0), (2,0), (2,2), and (0,1) -/
def quadI : List (ℝ × ℝ) := [(0,0), (2,0), (2,2), (0,1)]

/-- Quadrilateral II with vertices (0,0), (3,0), (3,1), and (0,2) -/
def quadII : List (ℝ × ℝ) := [(0,0), (3,0), (3,1), (0,2)]

theorem quadI_area_less_than_quadII_area :
  quadrilateralArea quadI.head! quadI.tail!.head! quadI.tail!.tail!.head! quadI.tail!.tail!.tail!.head! <
  quadrilateralArea quadII.head! quadII.tail!.head! quadII.tail!.tail!.head! quadII.tail!.tail!.tail!.head! :=
by sorry

end NUMINAMATH_CALUDE_quadI_area_less_than_quadII_area_l1774_177432


namespace NUMINAMATH_CALUDE_max_value_abc_inverse_sum_cubed_l1774_177474

theorem max_value_abc_inverse_sum_cubed (a b c : ℝ) (h : a + b + c = 0) :
  abc * (1/a + 1/b + 1/c)^3 ≤ 27/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_inverse_sum_cubed_l1774_177474


namespace NUMINAMATH_CALUDE_money_sum_l1774_177479

/-- Given three people A, B, and C with a total amount of money, prove the sum of A and C's money. -/
theorem money_sum (total money_B_C money_C : ℕ) 
  (h1 : total = 900)
  (h2 : money_B_C = 750)
  (h3 : money_C = 250) :
  ∃ (money_A : ℕ), money_A + money_C = 400 :=
by sorry

end NUMINAMATH_CALUDE_money_sum_l1774_177479


namespace NUMINAMATH_CALUDE_seven_people_round_table_l1774_177419

/-- The number of distinct arrangements of n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: There are 720 distinct ways to arrange 7 people around a round table. -/
theorem seven_people_round_table : roundTableArrangements 7 = 720 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l1774_177419


namespace NUMINAMATH_CALUDE_euler_triangle_inequality_l1774_177488

/-- 
For any triangle, let:
  r : radius of the incircle
  R : radius of the circumcircle
  d : distance between the incenter and circumcenter

Then, R ≥ 2r
-/
theorem euler_triangle_inequality (r R d : ℝ) : r > 0 → R > 0 → d > 0 → R ≥ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_euler_triangle_inequality_l1774_177488


namespace NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1774_177495

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1774_177495


namespace NUMINAMATH_CALUDE_no_inscribed_parallelepiped_l1774_177408

theorem no_inscribed_parallelepiped (π : ℝ) (h_π : π = Real.pi) :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x * y * z = 2 * π / 3 ∧
    x * y + y * z + z * x = π ∧
    x^2 + y^2 + z^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_parallelepiped_l1774_177408


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1774_177428

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first smaller triangle -/
  area1 : ℝ
  /-- Area of the second smaller triangle -/
  area2 : ℝ
  /-- Area of the third smaller triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ
  /-- The sum of all areas equals the area of the original triangle -/
  area_sum : area1 + area2 + area3 + areaQuad > 0

/-- The main theorem about the area of the quadrilateral -/
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) (h2 : t.area2 = 9) (h3 : t.area3 = 9) : 
  t.areaQuad = 45 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l1774_177428


namespace NUMINAMATH_CALUDE_prob_sum_three_eq_one_over_216_l1774_177485

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're looking for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three standard dice -/
def prob_sum_three : ℚ := (prob_single_die) ^ num_dice

theorem prob_sum_three_eq_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_eq_one_over_216_l1774_177485
