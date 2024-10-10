import Mathlib

namespace mrs_hilt_pencils_l550_55061

/-- The number of pencils Mrs. Hilt can buy -/
def pencils_bought (total_money : ℕ) (cost_per_pencil : ℕ) : ℕ :=
  total_money / cost_per_pencil

/-- Proof that Mrs. Hilt can buy 10 pencils -/
theorem mrs_hilt_pencils :
  pencils_bought 50 5 = 10 := by
  sorry

end mrs_hilt_pencils_l550_55061


namespace quantities_total_l550_55088

theorem quantities_total (total_avg : ℝ) (subset1_avg : ℝ) (subset2_avg : ℝ) 
  (h1 : total_avg = 8)
  (h2 : subset1_avg = 4)
  (h3 : subset2_avg = 14)
  (h4 : 3 * subset1_avg + 2 * subset2_avg = 5 * total_avg) : 
  5 = (3 * subset1_avg + 2 * subset2_avg) / total_avg :=
by sorry

end quantities_total_l550_55088


namespace circle_radius_from_area_circumference_ratio_l550_55067

/-- Given a circle with area A and circumference C, if A/C = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (A C : ℝ) (h : A / C = 15) :
  ∃ (r : ℝ), A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_from_area_circumference_ratio_l550_55067


namespace edward_money_problem_l550_55035

/-- Proves that if a person spends $17, then receives $10, and ends up with $7, they must have started with $14. -/
theorem edward_money_problem (initial_amount spent received final_amount : ℤ) :
  spent = 17 →
  received = 10 →
  final_amount = 7 →
  initial_amount - spent + received = final_amount →
  initial_amount = 14 :=
by sorry

end edward_money_problem_l550_55035


namespace inequality_solution_set_l550_55047

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x / (x - 1) < 1 - a ∧ x ≠ 1}
  (a > 0 → S = Set.Ioo ((a - 1) / a) 1) ∧
  (a = 0 → S = Set.Iio 1) ∧
  (a < 0 → S = Set.Iio 1 ∪ Set.Ioi ((a - 1) / a)) :=
by sorry

end inequality_solution_set_l550_55047


namespace imaginary_power_sum_l550_55053

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^17 + i^203 = 0 := by
  sorry

end imaginary_power_sum_l550_55053


namespace red_balls_count_l550_55085

theorem red_balls_count (white_balls : ℕ) (ratio_white : ℕ) (ratio_red : ℕ) : 
  white_balls = 16 → ratio_white = 4 → ratio_red = 3 → 
  (white_balls * ratio_red) / ratio_white = 12 := by
sorry

end red_balls_count_l550_55085


namespace soda_price_ratio_l550_55096

/-- Represents the volume and price of a soda brand relative to Brand Y -/
structure SodaBrand where
  volume : ℚ  -- Relative volume compared to Brand Y
  price : ℚ   -- Relative price compared to Brand Y

/-- Calculates the unit price of a soda brand -/
def unitPrice (brand : SodaBrand) : ℚ :=
  brand.price / brand.volume

theorem soda_price_ratio :
  let brand_x : SodaBrand := { volume := 13/10, price := 17/20 }
  let brand_z : SodaBrand := { volume := 14/10, price := 11/10 }
  (unitPrice brand_z) / (unitPrice brand_x) = 13/11 := by
  sorry

end soda_price_ratio_l550_55096


namespace expansion_simplification_l550_55076

theorem expansion_simplification (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x^2 + 5 * x^3 - 2 / 3) = 3 / x^2 + 15 * x^3 / 4 - 1 / 2 := by
  sorry

end expansion_simplification_l550_55076


namespace spatial_relationship_l550_55092

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (para_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem spatial_relationship 
  (m l : Line) 
  (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : perp m α) 
  (h4 : para l β) 
  (h5 : para_planes α β) : 
  perp_lines m l :=
sorry

end spatial_relationship_l550_55092


namespace parallelogram_height_calculation_l550_55006

/-- Given a parallelogram-shaped field with specified dimensions and costs, 
    calculate the perpendicular distance from the other side. -/
theorem parallelogram_height_calculation 
  (base : ℝ)
  (cost_per_10sqm : ℝ)
  (total_cost : ℝ)
  (h : base = 54)
  (i : cost_per_10sqm = 50)
  (j : total_cost = 6480) :
  (total_cost / cost_per_10sqm * 10) / base = 24 := by
sorry

end parallelogram_height_calculation_l550_55006


namespace lady_eagles_score_l550_55074

theorem lady_eagles_score (total_points : ℕ) (games : ℕ) (jessie_points : ℕ)
  (h1 : total_points = 311)
  (h2 : games = 5)
  (h3 : jessie_points = 41) :
  total_points - 3 * jessie_points = 188 := by
  sorry

end lady_eagles_score_l550_55074


namespace sector_area_l550_55009

/-- The area of a circular sector with central angle 240° and radius 6 is 24π -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 240 * π / 180 → r = 6 → (1/2) * r^2 * θ = 24 * π := by
  sorry

end sector_area_l550_55009


namespace odd_function_property_l550_55073

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (a b : ℝ) :
  let f := fun (x : ℝ) ↦ (a * x + b) / (x^2 + 1)
  IsOdd f ∧ f (1/2) = 2/5 → f 2 = 2/5 := by
  sorry

end odd_function_property_l550_55073


namespace subtraction_problem_l550_55083

theorem subtraction_problem (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end subtraction_problem_l550_55083


namespace salary_relation_l550_55023

theorem salary_relation (A B C : ℝ) :
  A + B + C = 10000 ∧
  A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
  0.1 * A + 0.15 * B = 0.2 * C →
  A = 20000 / 3 - 7 * B / 6 :=
by sorry

end salary_relation_l550_55023


namespace sequence_formulas_l550_55045

/-- Given an arithmetic sequence a_n with first term 19 and common difference -2,
    and a geometric sequence b_n - a_n with first term 1 and common ratio 3,
    prove the formulas for a_n, S_n, b_n, and T_n. -/
theorem sequence_formulas (n : ℕ) :
  let a : ℕ → ℝ := λ k => 19 - 2 * (k - 1)
  let S : ℕ → ℝ := λ k => (k * (a 1 + a k)) / 2
  let b : ℕ → ℝ := λ k => a k + 3^(k - 1)
  let T : ℕ → ℝ := λ k => S k + (3^k - 1) / 2
  (a n = 21 - 2 * n) ∧
  (S n = 20 * n - n^2) ∧
  (b n = 21 - 2 * n + 3^(n - 1)) ∧
  (T n = 20 * n - n^2 + (3^n - 1) / 2) :=
by sorry

end sequence_formulas_l550_55045


namespace solution_set_of_inequality_l550_55097

noncomputable def f (x : ℝ) : ℝ := x - (Real.exp 1 - 1) * Real.log x

theorem solution_set_of_inequality (x : ℝ) :
  (f (Real.exp x) < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_of_inequality_l550_55097


namespace function_range_l550_55032

theorem function_range (x : ℝ) : 
  (∀ a : ℝ, a ∈ Set.Icc (-1 : ℝ) 1 → 
    (a * x^2 - (2*a + 1) * x + a + 1 < 0)) → 
  (1 < x ∧ x < 2) := by
  sorry

end function_range_l550_55032


namespace computer_purchase_cost_effectiveness_l550_55044

def store_A_cost (x : ℕ) : ℝ := 4500 * x + 1500
def store_B_cost (x : ℕ) : ℝ := 4800 * x

theorem computer_purchase_cost_effectiveness (x : ℕ) :
  (x < 5 → store_B_cost x < store_A_cost x) ∧
  (x > 5 → store_A_cost x < store_B_cost x) ∧
  (x = 5 → store_A_cost x = store_B_cost x) := by
  sorry

end computer_purchase_cost_effectiveness_l550_55044


namespace z_120_20_bounds_l550_55049

/-- Z_{2k}^s is the s-th member from the center in the 2k-th row -/
def Z (k : ℕ) (s : ℕ) : ℝ := sorry

/-- w_{2k} is a function of k -/
def w (k : ℕ) : ℝ := sorry

/-- Main theorem: Z_{120}^{20} is bounded between 0.012 and 0.016 -/
theorem z_120_20_bounds :
  0.012 < Z 60 10 ∧ Z 60 10 < 0.016 :=
sorry

end z_120_20_bounds_l550_55049


namespace vector_collinearity_implies_x_value_l550_55021

theorem vector_collinearity_implies_x_value (x : ℝ) 
  (hx : x > 0) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (8, x/2))
  (hb : b = (x, 1))
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ (a - 2 • b) = k • (2 • a + b)) :
  x = 4 := by
sorry

end vector_collinearity_implies_x_value_l550_55021


namespace max_value_of_linear_combination_l550_55039

theorem max_value_of_linear_combination (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 8 → (∀ a b : ℝ, 4*x + 3*y ≤ 64) ∧ (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 8 ∧ 4*x₀ + 3*y₀ = 64) :=
by sorry

end max_value_of_linear_combination_l550_55039


namespace work_completion_time_l550_55072

/-- Given workers A and B, where A can finish a job in 4 days and B in 14 days,
    prove that after working together for 2 days and A leaving,
    B will take 5 more days to finish the job. -/
theorem work_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (days_together : ℝ) 
  (h1 : days_A = 4) 
  (h2 : days_B = 14) 
  (h3 : days_together = 2) : 
  (days_B - (1 - (days_together * (1 / days_A + 1 / days_B))) / (1 / days_B)) = 5 := by
  sorry

end work_completion_time_l550_55072


namespace road_repair_equivalence_l550_55059

/-- The number of persons in the first group -/
def first_group : ℕ := 36

/-- The number of days to complete the work -/
def days : ℕ := 12

/-- The number of hours worked per day by the first group -/
def hours_first : ℕ := 5

/-- The number of hours worked per day by the second group -/
def hours_second : ℕ := 6

/-- The number of persons in the second group -/
def second_group : ℕ := 30

theorem road_repair_equivalence :
  first_group * days * hours_first = second_group * days * hours_second :=
sorry

end road_repair_equivalence_l550_55059


namespace train_stop_time_l550_55051

/-- Proves that a train with given speeds including and excluding stoppages
    stops for 20 minutes per hour. -/
theorem train_stop_time
  (speed_without_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : speed_with_stops = 40)
  : (1 - speed_with_stops / speed_without_stops) * 60 = 20 :=
by sorry

end train_stop_time_l550_55051


namespace systematic_sample_selection_l550_55024

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is selected in a systematic sample -/
def is_selected (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.total

theorem systematic_sample_selection 
  (s : SystematicSample)
  (h_total : s.total = 900)
  (h_size : s.sample_size = 150)
  (h_start : s.start = 15)
  (h_interval : s.interval = s.total / s.sample_size)
  (h_15_selected : is_selected s 15)
  : is_selected s 81 := by
  sorry

end systematic_sample_selection_l550_55024


namespace complex_power_eight_l550_55050

theorem complex_power_eight :
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6))) ^ 8 =
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by
  sorry

end complex_power_eight_l550_55050


namespace calculation_proof_l550_55033

theorem calculation_proof :
  (1) * (Real.pi - 3.14) ^ 0 - |2 - Real.sqrt 3| + (-1/2)^2 = Real.sqrt 3 - 3/4 ∧
  Real.sqrt (1/3) + Real.sqrt 6 * (1/Real.sqrt 2 + Real.sqrt 8) = 16 * Real.sqrt 3 / 3 :=
by sorry

end calculation_proof_l550_55033


namespace rs_length_l550_55094

/-- Triangle PQR with point S on PR -/
structure TrianglePQR where
  /-- Length of PQ -/
  PQ : ℝ
  /-- Length of QR -/
  QR : ℝ
  /-- Length of PS -/
  PS : ℝ
  /-- Length of QS -/
  QS : ℝ
  /-- PQ equals QR -/
  PQ_eq_QR : PQ = QR
  /-- PQ equals 8 -/
  PQ_eq_8 : PQ = 8
  /-- PS equals 10 -/
  PS_eq_10 : PS = 10
  /-- QS equals 5 -/
  QS_eq_5 : QS = 5

/-- The length of RS in the given triangle configuration is 3.5 -/
theorem rs_length (t : TrianglePQR) : ∃ RS : ℝ, RS = 3.5 := by
  sorry

end rs_length_l550_55094


namespace right_triangle_cos_z_l550_55042

theorem right_triangle_cos_z (X Y Z : Real) (h1 : X + Y + Z = π) (h2 : X = π/2) (h3 : Real.sin Y = 3/5) :
  Real.cos Z = 3/5 := by
  sorry

end right_triangle_cos_z_l550_55042


namespace clock_angles_l550_55056

/-- Represents the angle between the hour and minute hands on a clock face -/
def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

/-- Definition of a straight angle -/
def isStraightAngle (angle : ℝ) : Prop :=
  angle = 180

/-- Definition of a right angle -/
def isRightAngle (angle : ℝ) : Prop :=
  angle = 90

/-- Definition of an obtuse angle -/
def isObtuseAngle (angle : ℝ) : Prop :=
  90 < angle ∧ angle < 180

theorem clock_angles :
  (isStraightAngle (clockAngle 6 0)) ∧
  (isRightAngle (clockAngle 9 0)) ∧
  (isObtuseAngle (clockAngle 4 0)) :=
by sorry

end clock_angles_l550_55056


namespace van_tire_mileage_l550_55071

/-- Calculates the miles each tire is used given the total miles traveled,
    number of tires, and number of tires used at a time. -/
def miles_per_tire (total_miles : ℕ) (num_tires : ℕ) (tires_in_use : ℕ) : ℚ :=
  (total_miles * tires_in_use : ℚ) / num_tires

/-- Proves that for a van with 7 tires, where 6 are used at a time,
    and the van travels 42,000 miles with all tires equally worn,
    each tire is used for 36,000 miles. -/
theorem van_tire_mileage :
  miles_per_tire 42000 7 6 = 36000 := by sorry

end van_tire_mileage_l550_55071


namespace solve_linear_equation_l550_55093

theorem solve_linear_equation (x : ℝ) :
  3 * x - 4 * x + 5 * x = 140 → x = 35 := by
  sorry

end solve_linear_equation_l550_55093


namespace baker_cakes_l550_55079

theorem baker_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) :
  initial ≥ sold →
  initial - sold + bought = initial + bought - sold :=
by sorry

end baker_cakes_l550_55079


namespace diagonal_length_isosceles_trapezoid_l550_55046

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ) -- longer base
  (CD : ℝ) -- shorter base
  (AD : ℝ) -- leg
  (BC : ℝ) -- leg
  (isIsosceles : AD = BC)
  (isPositive : AB > 0 ∧ CD > 0 ∧ AD > 0)
  (baseOrder : AB > CD)

-- Theorem statement
theorem diagonal_length_isosceles_trapezoid (T : IsoscelesTrapezoid) 
  (h1 : T.AB = 25) 
  (h2 : T.CD = 13) 
  (h3 : T.AD = 12) :
  Real.sqrt ((25 - 13) ^ 2 / 4 + (Real.sqrt (12 ^ 2 - ((25 - 13) / 2) ^ 2)) ^ 2) = 12 :=
by sorry

end diagonal_length_isosceles_trapezoid_l550_55046


namespace fruit_purchase_cost_l550_55014

/-- Calculates the total cost of fruits with a discount -/
def totalCostWithDiscount (cherryPrice olivePrice : ℚ) (bagCount : ℕ) (discountPercentage : ℚ) : ℚ :=
  let discountFactor : ℚ := 1 - discountPercentage / 100
  let discountedCherryPrice : ℚ := cherryPrice * discountFactor
  let discountedOlivePrice : ℚ := olivePrice * discountFactor
  (discountedCherryPrice + discountedOlivePrice) * bagCount

/-- Proves that the total cost for 50 bags each of cherries and olives with a 10% discount is $540 -/
theorem fruit_purchase_cost : 
  totalCostWithDiscount 5 7 50 10 = 540 := by
  sorry

end fruit_purchase_cost_l550_55014


namespace park_fencing_cost_l550_55013

/-- The cost of fencing one side of the square park -/
def cost_per_side : ℕ := 56

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing the square park -/
def total_cost : ℕ := cost_per_side * num_sides

theorem park_fencing_cost : total_cost = 224 := by
  sorry

end park_fencing_cost_l550_55013


namespace other_communities_count_l550_55043

theorem other_communities_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 46 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 136 ∧ 
    (↑other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end other_communities_count_l550_55043


namespace x_varies_as_four_thirds_power_of_z_l550_55095

/-- Given that x varies as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := by
sorry

end x_varies_as_four_thirds_power_of_z_l550_55095


namespace division_remainder_l550_55037

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 222 → divisor = 13 → quotient = 17 → 
  dividend = divisor * quotient + remainder → remainder = 1 := by
sorry

end division_remainder_l550_55037


namespace milburg_children_count_l550_55012

/-- The number of children in Milburg -/
def children_count (total_population grown_ups : ℕ) : ℕ :=
  total_population - grown_ups

/-- Theorem stating the number of children in Milburg -/
theorem milburg_children_count :
  children_count 8243 5256 = 2987 := by
  sorry

end milburg_children_count_l550_55012


namespace limit_of_sequence_a_l550_55086

def a (n : ℕ) : ℚ := (2 * n - 1) / (2 - 3 * n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-2/3)| < ε :=
sorry

end limit_of_sequence_a_l550_55086


namespace triangle_properties_l550_55031

/-- Given a triangle ABC with A = 2B, b = 2, and c = 4, prove the following properties -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_A_2B : A = 2 * B)
  (h_b : b = 2)
  (h_c : c = 4) :
  a = 2 * b * Real.cos B ∧ B = π / 6 := by
  sorry

end triangle_properties_l550_55031


namespace weight_of_doubled_cube_l550_55002

/-- Given a cube of metal weighing 6 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 48 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ) (h1 : weight = 6) :
  let new_weight := weight * (2^3)
  new_weight = 48 := by sorry

end weight_of_doubled_cube_l550_55002


namespace smaller_number_proof_l550_55016

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  min a b = 25 := by
  sorry

end smaller_number_proof_l550_55016


namespace triangle_inequality_l550_55063

/-- For any triangle ABC with sides a, b, and c, the sum of squares of the sides
    is greater than or equal to 4√3 times the area of the triangle. -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let S := Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by sorry

end triangle_inequality_l550_55063


namespace linear_arrangement_paths_count_l550_55000

/-- Represents a linear arrangement of nodes -/
structure LinearArrangement (n : ℕ) where
  nodes : Fin n → ℕ

/-- Counts the number of paths of a given length in a linear arrangement -/
def countPaths (arr : LinearArrangement 10) (length : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of paths of length 4 in a linear arrangement of 10 nodes is 2304 -/
theorem linear_arrangement_paths_count :
  ∀ (arr : LinearArrangement 10), countPaths arr 4 = 2304 := by
  sorry

end linear_arrangement_paths_count_l550_55000


namespace parametric_equations_form_circle_parametric_equations_part_of_circle_l550_55082

noncomputable def parametricCircle (θ : Real) : Real × Real :=
  (4 - Real.cos θ, 1 - Real.sin θ)

theorem parametric_equations_form_circle (θ : Real) 
  (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) : 
  let (x, y) := parametricCircle θ
  (x - 4)^2 + (y - 1)^2 = 1 := by
sorry

theorem parametric_equations_part_of_circle :
  ∃ (a b r : Real), 
    (∀ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
      let (x, y) := parametricCircle θ
      (x - a)^2 + (y - b)^2 = r^2) ∧
    (∃ θ₁ θ₂, 0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ Real.pi / 2 ∧ 
      parametricCircle θ₁ ≠ parametricCircle θ₂) := by
sorry

end parametric_equations_form_circle_parametric_equations_part_of_circle_l550_55082


namespace remaining_jelly_beans_l550_55078

/-- Represents the distribution of jelly beans based on ID endings -/
structure JellyBeanDistribution :=
  (group1 : Nat) (group2 : Nat) (group3 : Nat)
  (group4 : Nat) (group5 : Nat) (group6 : Nat)

/-- Calculates the total number of jelly beans drawn -/
def totalJellyBeansDrawn (dist : JellyBeanDistribution) : Nat :=
  dist.group1 * 2 + dist.group2 * 4 + dist.group3 * 6 +
  dist.group4 * 8 + dist.group5 * 10 + dist.group6 * 12

/-- Theorem stating the number of remaining jelly beans -/
theorem remaining_jelly_beans
  (initial_jelly_beans : Nat)
  (total_children : Nat)
  (allowed_percentage : Rat)
  (dist : JellyBeanDistribution) :
  initial_jelly_beans = 2000 →
  total_children = 100 →
  allowed_percentage = 70 / 100 →
  dist.group1 = 9 →
  dist.group2 = 25 →
  dist.group3 = 20 →
  dist.group4 = 15 →
  dist.group5 = 15 →
  dist.group6 = 14 →
  initial_jelly_beans - totalJellyBeansDrawn dist = 1324 := by
  sorry

end remaining_jelly_beans_l550_55078


namespace negative_765_degrees_conversion_l550_55038

theorem negative_765_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    (-765 : ℝ) * π / 180 = 2 * k * π + α ∧ 
    0 ≤ α ∧ 
    α < 2 * π ∧ 
    k = -3 ∧ 
    α = 7 * π / 4 := by
  sorry

end negative_765_degrees_conversion_l550_55038


namespace number_of_hydroxide_groups_l550_55058

/-- The atomic weight of aluminum -/
def atomic_weight_Al : ℝ := 27

/-- The molecular weight of a hydroxide group -/
def molecular_weight_OH : ℝ := 17

/-- The molecular weight of the compound Al(OH)n -/
def molecular_weight_compound : ℝ := 78

/-- The number of hydroxide groups in the compound -/
def n : ℕ := sorry

/-- Theorem stating that the number of hydroxide groups in Al(OH)n is 3 -/
theorem number_of_hydroxide_groups :
  n = 3 :=
sorry

end number_of_hydroxide_groups_l550_55058


namespace root_in_interval_l550_55091

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 1.5 > 0) → ∃ x, x ∈ Set.Ioo 1 1.5 ∧ f x = 0 := by
  sorry

end root_in_interval_l550_55091


namespace hyperbola_equation_l550_55077

/-- Hyperbola with given properties -/
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧
  (∀ x : ℝ, ∃ y : ℝ, y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) ∧
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |Real.sqrt 3 * x - 3 * y| / Real.sqrt 12 = 1)

/-- The equation of the hyperbola with the given properties -/
theorem hyperbola_equation :
  ∀ a b : ℝ, Hyperbola a b → (∀ x y : ℝ, x^2 / 4 - 3 * y^2 / 4 = 1) :=
by sorry

end hyperbola_equation_l550_55077


namespace hyperbola_asymptote_l550_55007

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (hyperbola x y) → (asymptote x y) :=
by sorry

end hyperbola_asymptote_l550_55007


namespace sum_modulo_thirteen_l550_55001

theorem sum_modulo_thirteen : (9375 + 9376 + 9377 + 9378 + 9379) % 13 = 7 := by
  sorry

end sum_modulo_thirteen_l550_55001


namespace bus_driver_average_hours_l550_55025

/-- The average number of hours the bus driver drives each day -/
def average_hours : ℝ := 2

/-- The average speed from Monday to Wednesday in km/h -/
def speed_mon_wed : ℝ := 12

/-- The average speed from Thursday to Friday in km/h -/
def speed_thu_fri : ℝ := 9

/-- The total distance traveled in 5 days in km -/
def total_distance : ℝ := 108

/-- The number of days driven from Monday to Wednesday -/
def days_mon_wed : ℝ := 3

/-- The number of days driven from Thursday to Friday -/
def days_thu_fri : ℝ := 2

theorem bus_driver_average_hours :
  average_hours * speed_mon_wed * days_mon_wed +
  average_hours * speed_thu_fri * days_thu_fri = total_distance :=
sorry

end bus_driver_average_hours_l550_55025


namespace evaluate_expression_l550_55010

theorem evaluate_expression : 5 - 7 * (8 - 12 / (3^2)) * 6 = -275 := by
  sorry

end evaluate_expression_l550_55010


namespace spinner_ice_cream_prices_l550_55022

-- Define the price of a spinner and an ice cream
variable (s m : ℝ)

-- Define Petya's and Vasya's claims
def petya_claim := 2 * s > 5 * m
def vasya_claim := 3 * s > 8 * m

-- Theorem statement
theorem spinner_ice_cream_prices 
  (h1 : (petya_claim s m ∧ ¬vasya_claim s m) ∨ (¬petya_claim s m ∧ vasya_claim s m))
  (h2 : vasya_claim s m) :
  7 * s ≤ 19 * m := by
  sorry

end spinner_ice_cream_prices_l550_55022


namespace completing_square_result_l550_55041

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end completing_square_result_l550_55041


namespace hulk_jumps_theorem_l550_55069

def jump_sequence (n : ℕ) : ℝ := 4 * (3 : ℝ) ^ (n - 1)

def total_distance (n : ℕ) : ℝ := 2 * ((3 : ℝ) ^ n - 1)

theorem hulk_jumps_theorem :
  (∀ k < 8, total_distance k ≤ 5000) ∧ total_distance 8 > 5000 := by
  sorry

end hulk_jumps_theorem_l550_55069


namespace binomial_distribution_parameters_l550_55052

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_parameters :
  ∃ (ξ : BinomialDistribution), expectedValue ξ = 12 ∧ variance ξ = 2.4 ∧ ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end binomial_distribution_parameters_l550_55052


namespace relationship_correctness_l550_55005

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1/a < 1/b) → False) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∃ a b c : ℝ, (a > b ∧ b > 0 → a^c < b^c) → False) :=
by sorry

end relationship_correctness_l550_55005


namespace custom_op_three_six_l550_55020

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end custom_op_three_six_l550_55020


namespace solve_for_a_l550_55080

theorem solve_for_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 20 - 3*a) : a = 20/7 := by
  sorry

end solve_for_a_l550_55080


namespace cylinder_radius_and_volume_l550_55065

/-- Properties of a cylinder with given height and surface area -/
def Cylinder (h : ℝ) (s : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ h > 0 ∧ s = 2 * Real.pi * r * h + 2 * Real.pi * r^2

theorem cylinder_radius_and_volume 
  (h : ℝ) (s : ℝ) 
  (hh : h = 8) (hs : s = 130 * Real.pi) : 
  ∃ (r v : ℝ), Cylinder h s ∧ r = 5 ∧ v = 200 * Real.pi := by
sorry

end cylinder_radius_and_volume_l550_55065


namespace largest_prime_divisor_of_17_squared_plus_144_squared_l550_55066

theorem largest_prime_divisor_of_17_squared_plus_144_squared :
  (Nat.factors (17^2 + 144^2)).maximum? = some 29 := by
  sorry

end largest_prime_divisor_of_17_squared_plus_144_squared_l550_55066


namespace cos_alpha_value_l550_55081

/-- Proves that for an angle α in the second quadrant, 
    if 2sin(2α) = cos(2α) - 1, then cos(α) = -√5/5 -/
theorem cos_alpha_value (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : 2 * Real.sin (2 * α) = Real.cos (2 * α) - 1) -- given equation
  : Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end cos_alpha_value_l550_55081


namespace det_specific_matrix_l550_55030

theorem det_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end det_specific_matrix_l550_55030


namespace min_M_and_F_M_l550_55026

def is_k_multiple (n : ℕ) (k : ℤ) : Prop :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  (thousands + hundreds : ℤ) = k * (tens - units)

def swap_hundreds_tens (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  thousands * 1000 + tens * 100 + hundreds * 10 + units

def F (m : ℕ) : ℚ :=
  let a : ℕ := (m + 1) / 2
  let b : ℕ := (m - 1) / 2
  (a : ℚ) / b

theorem min_M_and_F_M :
  ∃ (M : ℕ),
    M ≥ 1000 ∧ M < 10000 ∧
    is_k_multiple M 4 ∧
    is_k_multiple (M - 4) (-3) ∧
    is_k_multiple (swap_hundreds_tens M) 4 ∧
    (∀ (N : ℕ), N ≥ 1000 ∧ N < 10000 ∧
      is_k_multiple N 4 ∧
      is_k_multiple (N - 4) (-3) ∧
      is_k_multiple (swap_hundreds_tens N) 4 →
      M ≤ N) ∧
    M = 6663 ∧
    F M = 3332 / 3331 := by sorry

end min_M_and_F_M_l550_55026


namespace gcd_lcm_sum_l550_55017

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by
  sorry

end gcd_lcm_sum_l550_55017


namespace dorothy_income_l550_55070

theorem dorothy_income (annual_income : ℝ) : 
  annual_income * (1 - 0.18) = 49200 → annual_income = 60000 := by
  sorry

end dorothy_income_l550_55070


namespace fraction_sum_l550_55004

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_sum_l550_55004


namespace solution_set_characterization_l550_55090

/-- The set of real numbers a for which the system of equations has at least one solution -/
def SolutionSet : Set ℝ :=
  {a | ∃ x y, x - 1 = a * (y^3 - 1) ∧
               2 * x / (|y^3| + y^3) = Real.sqrt x ∧
               y > 0 ∧
               x ≥ 0}

/-- Theorem stating that the SolutionSet is equal to the union of three intervals -/
theorem solution_set_characterization :
  SolutionSet = {a | a < 0} ∪ {a | 0 ≤ a ∧ a ≤ 1} ∪ {a | a > 1} :=
by sorry

end solution_set_characterization_l550_55090


namespace prove_research_paper_requirement_l550_55034

def research_paper_requirement (yvonne_words janna_extra_words removed_words added_multiplier additional_words : ℕ) : Prop :=
  let janna_words := yvonne_words + janna_extra_words
  let initial_total := yvonne_words + janna_words
  let after_removal := initial_total - removed_words
  let added_words := removed_words * added_multiplier
  let after_addition := after_removal + added_words
  let final_requirement := after_addition + additional_words
  final_requirement = 1000

theorem prove_research_paper_requirement :
  research_paper_requirement 400 150 20 2 30 := by
  sorry

end prove_research_paper_requirement_l550_55034


namespace orange_crates_pigeonhole_l550_55057

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 7 ∧ 
    ∃ k : ℕ, min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ subset : Finset (Fin total_crates), subset.card = n ∧
        ∀ i ∈ subset, ∃ f : Fin total_crates → ℕ, 
          (∀ j, min_oranges ≤ f j ∧ f j ≤ max_oranges) ∧ f i = k) :=
by sorry

end orange_crates_pigeonhole_l550_55057


namespace initial_onions_l550_55098

theorem initial_onions (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 33) :
  sold + left = 98 := by
  sorry

end initial_onions_l550_55098


namespace monotone_decreasing_implies_a_leq_neg_seven_l550_55003

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + 2

-- State the theorem
theorem monotone_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a y < f a x) →
  a ≤ -7 := by
  sorry

end monotone_decreasing_implies_a_leq_neg_seven_l550_55003


namespace artist_painting_rate_l550_55099

/-- Proves that given the specified conditions, the artist can paint 1.5 square meters per hour -/
theorem artist_painting_rate 
  (mural_length : ℝ) 
  (mural_width : ℝ) 
  (paint_cost_per_sqm : ℝ) 
  (artist_hourly_rate : ℝ) 
  (total_mural_cost : ℝ) 
  (h1 : mural_length = 6) 
  (h2 : mural_width = 3) 
  (h3 : paint_cost_per_sqm = 4) 
  (h4 : artist_hourly_rate = 10) 
  (h5 : total_mural_cost = 192) : 
  (mural_length * mural_width) / ((total_mural_cost - (paint_cost_per_sqm * mural_length * mural_width)) / artist_hourly_rate) = 1.5 := by
  sorry

end artist_painting_rate_l550_55099


namespace ab_equality_l550_55019

theorem ab_equality (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end ab_equality_l550_55019


namespace nonCoplanarChoices_eq_141_l550_55027

/-- The number of ways to choose 4 non-coplanar points from the vertices and midpoints of a tetrahedron -/
def nonCoplanarChoices : ℕ :=
  Nat.choose 10 4 - (4 * Nat.choose 6 4 + 6 + 3)

/-- Theorem stating that the number of ways to choose 4 non-coplanar points
    from the vertices and midpoints of a tetrahedron is 141 -/
theorem nonCoplanarChoices_eq_141 : nonCoplanarChoices = 141 := by
  sorry

end nonCoplanarChoices_eq_141_l550_55027


namespace intersection_M_N_l550_55064

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end intersection_M_N_l550_55064


namespace power_of_product_l550_55048

theorem power_of_product (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end power_of_product_l550_55048


namespace polynomial_characterization_l550_55028

/-- A homogeneous polynomial of degree n in two variables -/
noncomputable def HomogeneousPolynomial (n : ℕ) := (ℝ → ℝ → ℝ)

/-- The property of being homogeneous of degree n -/
def IsHomogeneous (P : HomogeneousPolynomial n) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

/-- The second condition from the problem -/
def SatisfiesCondition2 (P : HomogeneousPolynomial n) : Prop :=
  ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The third condition from the problem -/
def SatisfiesCondition3 (P : HomogeneousPolynomial n) : Prop :=
  P 1 0 = 1

/-- The theorem statement -/
theorem polynomial_characterization (n : ℕ) (P : HomogeneousPolynomial n)
  (h1 : IsHomogeneous P)
  (h2 : SatisfiesCondition2 P)
  (h3 : SatisfiesCondition3 P) :
  ∀ (x y : ℝ), P x y = (x + y)^(n - 1) * (x - 2*y) :=
sorry

end polynomial_characterization_l550_55028


namespace largest_root_is_four_l550_55036

/-- The polynomial function representing the difference between the curve and the line -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^6 - 10*x^5 + 29*x^4 - 4*x^3 + a*x^2 - b*x - c

/-- The statement that the polynomial has exactly three distinct roots, each with multiplicity 2 -/
def has_three_double_roots (a b c : ℝ) : Prop :=
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    ∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)

/-- The theorem stating that under the given conditions, 4 is the largest root -/
theorem largest_root_is_four (a b c : ℝ) (h : has_three_double_roots a b c) :
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)) ∧
    4 = max p (max q r) :=
  sorry

end largest_root_is_four_l550_55036


namespace pencil_count_l550_55015

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 4 →
  pencils = 24 :=
by
  sorry

end pencil_count_l550_55015


namespace line_quadrants_m_range_l550_55018

theorem line_quadrants_m_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m - 2) * x + m → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) → 
  0 < m ∧ m < 2 := by
sorry

end line_quadrants_m_range_l550_55018


namespace equation_solutions_l550_55029

theorem equation_solutions (x : ℝ) : 
  x ≠ -2 → 
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48) ↔ 
  (x = 1.2 ∨ x = -81.2) := by
sorry

end equation_solutions_l550_55029


namespace partition_equality_l550_55008

/-- The number of partitions of n into non-negative powers of 2 -/
def b (n : ℕ) : ℕ := sorry

/-- The number of partitions of n which include at least one of every power of 2 
    from 1 up to the highest power of 2 in the partition -/
def c (n : ℕ) : ℕ := sorry

/-- For any non-negative integer n, b(n+1) = 2c(n) -/
theorem partition_equality (n : ℕ) : b (n + 1) = 2 * c n := by sorry

end partition_equality_l550_55008


namespace chess_tournament_participants_l550_55089

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1) / 2 = 153) → n = 18 := by
  sorry

end chess_tournament_participants_l550_55089


namespace arithmetic_sequence_common_difference_l550_55068

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 7) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end arithmetic_sequence_common_difference_l550_55068


namespace ceiling_floor_expression_l550_55011

theorem ceiling_floor_expression : 
  ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ - ⌈(2:ℝ)/3⌉ = -1 := by sorry

end ceiling_floor_expression_l550_55011


namespace popcorn_shrimp_orders_l550_55075

/-- Proves that the number of popcorn shrimp orders is 9 given the conditions -/
theorem popcorn_shrimp_orders 
  (catfish_cost : ℝ) 
  (shrimp_cost : ℝ) 
  (total_orders : ℕ) 
  (total_amount : ℝ) 
  (h1 : catfish_cost = 6)
  (h2 : shrimp_cost = 3.5)
  (h3 : total_orders = 26)
  (h4 : total_amount = 133.5) :
  ∃ (catfish_orders shrimp_orders : ℕ), 
    catfish_orders + shrimp_orders = total_orders ∧ 
    catfish_cost * (catfish_orders : ℝ) + shrimp_cost * (shrimp_orders : ℝ) = total_amount ∧
    shrimp_orders = 9 := by
  sorry

end popcorn_shrimp_orders_l550_55075


namespace intersection_implies_a_value_l550_55060

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∃ (a : ℝ), A a ∩ B a = {9} → a = -3 := by sorry

end intersection_implies_a_value_l550_55060


namespace hockey_team_ties_l550_55087

theorem hockey_team_ties (wins ties : ℕ) : 
  wins = ties + 12 →
  2 * wins + ties = 60 →
  ties = 12 := by
sorry

end hockey_team_ties_l550_55087


namespace starting_team_combinations_l550_55062

/-- The number of members in the water polo team -/
def team_size : ℕ := 18

/-- The number of players in the starting team -/
def starting_team_size : ℕ := 7

/-- The number of interchangeable positions -/
def interchangeable_positions : ℕ := 5

/-- The number of ways to choose the starting team -/
def choose_starting_team : ℕ := team_size * (team_size - 1) * (Nat.choose (team_size - 2) interchangeable_positions)

theorem starting_team_combinations :
  choose_starting_team = 1338176 := by
  sorry

end starting_team_combinations_l550_55062


namespace largest_angle_of_triangle_l550_55084

/-- Given a triangle PQR with side lengths p, q, and r satisfying certain conditions,
    prove that its largest angle is 120 degrees. -/
theorem largest_angle_of_triangle (p q r : ℝ) (h1 : p + 3*q + 3*r = p^2) (h2 : p + 3*q - 3*r = -1) :
  ∃ (P Q R : ℝ), 
    P + Q + R = 180 ∧ 
    0 < P ∧ 0 < Q ∧ 0 < R ∧
    P ≤ 120 ∧ Q ≤ 120 ∧ R = 120 :=
sorry

end largest_angle_of_triangle_l550_55084


namespace inequality_proof_l550_55055

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_proof_l550_55055


namespace solve_equation_l550_55040

theorem solve_equation : ∃ y : ℚ, 2*y + 3*y = 500 - (4*y + 6*y) → y = 100/3 := by
  sorry

end solve_equation_l550_55040


namespace chessboard_coverage_impossible_l550_55054

/-- Represents the type of L-shaped block -/
inductive LBlockType
  | Type1  -- Covers 3 white squares and 1 black square
  | Type2  -- Covers 3 black squares and 1 white square

/-- Represents the chessboard coverage problem -/
def ChessboardCoverage (n m : ℕ) (square_blocks : ℕ) (l_blocks : ℕ) : Prop :=
  ∃ (x : ℕ),
    -- Total number of white squares covered
    square_blocks * 2 + 3 * x + 1 * (l_blocks - x) = n * m / 2 ∧
    -- Total number of black squares covered
    square_blocks * 2 + 1 * x + 3 * (l_blocks - x) = n * m / 2 ∧
    -- x is the number of Type1 L-blocks, and should not exceed total L-blocks
    x ≤ l_blocks

/-- Theorem stating the impossibility of covering the 18x8 chessboard -/
theorem chessboard_coverage_impossible :
  ¬ ChessboardCoverage 18 8 9 7 :=
sorry

end chessboard_coverage_impossible_l550_55054
