import Mathlib

namespace rahul_share_l1609_160931

/-- Calculates the share of payment for a worker given the total payment and the time taken by both workers to complete the job individually --/
def calculateShare (totalPayment : ℚ) (worker1Time : ℚ) (worker2Time : ℚ) : ℚ :=
  let worker1Rate := 1 / worker1Time
  let worker2Rate := 1 / worker2Time
  let totalRate := worker1Rate + worker2Rate
  let worker1Share := worker1Rate / totalRate
  worker1Share * totalPayment

/-- Proves that Rahul's share of the payment is $42 given the specified conditions --/
theorem rahul_share :
  let rahulTime := 3
  let rajeshTime := 2
  let totalPayment := 105
  calculateShare totalPayment rahulTime rajeshTime = 42 := by
  sorry

#eval calculateShare 105 3 2

end rahul_share_l1609_160931


namespace tmall_transaction_scientific_notation_l1609_160910

theorem tmall_transaction_scientific_notation :
  let transaction_volume : ℝ := 2135 * 10^9
  transaction_volume = 2.135 * 10^11 := by
  sorry

end tmall_transaction_scientific_notation_l1609_160910


namespace not_perfect_square_l1609_160991

theorem not_perfect_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
  ¬ ∃ k : ℕ, 3^m + 3^n + 1 = k^2 := by
sorry

end not_perfect_square_l1609_160991


namespace min_distance_to_line_l1609_160999

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ 8 * x₀ + 15 * y₀ = 120 ∧
  (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → 8 * x' + 15 * y' = 120 → 
    Real.sqrt (x₀^2 + y₀^2) ≤ Real.sqrt (x'^2 + y'^2)) ∧
  Real.sqrt (x₀^2 + y₀^2) = 120 / 17 :=
sorry

end min_distance_to_line_l1609_160999


namespace exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l1609_160965

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : exterior_angle_regular_octagon = 45 := by
  sorry

end exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l1609_160965


namespace jim_reading_pages_l1609_160992

/-- Calculates the number of pages Jim reads per week after changing his reading speed and time --/
def pages_read_per_week (
  regular_rate : ℝ)
  (technical_rate : ℝ)
  (regular_time : ℝ)
  (technical_time : ℝ)
  (regular_speed_increase : ℝ)
  (technical_speed_increase : ℝ)
  (regular_time_reduction : ℝ)
  (technical_time_reduction : ℝ) : ℝ :=
  let new_regular_rate := regular_rate * regular_speed_increase
  let new_technical_rate := technical_rate * technical_speed_increase
  let new_regular_time := regular_time - regular_time_reduction
  let new_technical_time := technical_time - technical_time_reduction
  (new_regular_rate * new_regular_time) + (new_technical_rate * new_technical_time)

theorem jim_reading_pages : 
  pages_read_per_week 40 30 10 5 1.5 1.3 4 2 = 477 := by
  sorry

end jim_reading_pages_l1609_160992


namespace wall_width_l1609_160990

/-- Given a rectangular wall with specific proportions and volume, prove its width is 4 meters. -/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 16128) : w = 4 := by
  sorry

end wall_width_l1609_160990


namespace training_schedule_days_l1609_160994

/-- Calculates the number of days required to complete a training schedule. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that the training schedule takes 10 days to complete. -/
theorem training_schedule_days :
  trainingDays 5 10 20 = 10 := by
  sorry

#eval trainingDays 5 10 20

end training_schedule_days_l1609_160994


namespace max_pieces_3x3_cake_l1609_160914

/-- Represents a rectangular cake -/
structure Cake where
  rows : ℕ
  cols : ℕ

/-- Represents a straight cut on the cake -/
structure Cut where
  max_intersections : ℕ

/-- Calculates the maximum number of pieces after one cut -/
def max_pieces_after_cut (cake : Cake) (cut : Cut) : ℕ :=
  2 * cut.max_intersections + 4

/-- Theorem: For a 3x3 cake, the maximum number of pieces after one cut is 14 -/
theorem max_pieces_3x3_cake (cake : Cake) (cut : Cut) :
  cake.rows = 3 ∧ cake.cols = 3 ∧ cut.max_intersections = 5 →
  max_pieces_after_cut cake cut = 14 := by
  sorry

end max_pieces_3x3_cake_l1609_160914


namespace asphalt_cost_asphalt_cost_proof_l1609_160904

/-- Calculates the total cost of asphalt for paving a road, including sales tax. -/
theorem asphalt_cost (road_length : ℝ) (road_width : ℝ) (coverage_per_truckload : ℝ) 
  (cost_per_truckload : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let road_area := road_length * road_width
  let num_truckloads := road_area / coverage_per_truckload
  let total_cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_with_tax := total_cost_before_tax + sales_tax
  total_cost_with_tax

/-- Proves that the total cost of asphalt for the given road specifications is $4,500. -/
theorem asphalt_cost_proof :
  asphalt_cost 2000 20 800 75 0.2 = 4500 := by
  sorry

end asphalt_cost_asphalt_cost_proof_l1609_160904


namespace unique_prime_satisfying_condition_l1609_160960

theorem unique_prime_satisfying_condition : 
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end unique_prime_satisfying_condition_l1609_160960


namespace hot_day_price_correct_l1609_160939

/-- Represents the lemonade stand operation --/
structure LemonadeStand where
  totalDays : ℕ
  hotDays : ℕ
  cupsPerDay : ℕ
  costPerCup : ℚ
  totalProfit : ℚ
  hotDayPriceIncrease : ℚ

/-- Calculates the price of a cup on a hot day --/
def hotDayPrice (stand : LemonadeStand) : ℚ :=
  let regularPrice := (stand.totalProfit + stand.totalDays * stand.cupsPerDay * stand.costPerCup) /
    (stand.cupsPerDay * (stand.totalDays + stand.hotDays * stand.hotDayPriceIncrease))
  regularPrice * (1 + stand.hotDayPriceIncrease)

/-- Theorem stating that the hot day price is correct --/
theorem hot_day_price_correct (stand : LemonadeStand) : 
  stand.totalDays = 10 ∧ 
  stand.hotDays = 4 ∧ 
  stand.cupsPerDay = 32 ∧ 
  stand.costPerCup = 3/4 ∧ 
  stand.totalProfit = 200 ∧
  stand.hotDayPriceIncrease = 1/4 →
  hotDayPrice stand = 25/16 := by
  sorry

#eval hotDayPrice {
  totalDays := 10
  hotDays := 4
  cupsPerDay := 32
  costPerCup := 3/4
  totalProfit := 200
  hotDayPriceIncrease := 1/4
}

end hot_day_price_correct_l1609_160939


namespace evaluate_expression_l1609_160944

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : 2*x - b + 5 = b + 23 := by
  sorry

end evaluate_expression_l1609_160944


namespace power_multiplication_l1609_160912

theorem power_multiplication (n : ℕ) :
  3000 * (3000 ^ 3000) = 3000 ^ (3000 + 1) :=
by sorry

end power_multiplication_l1609_160912


namespace scores_analysis_l1609_160930

def scores : List ℕ := [7, 5, 9, 7, 4, 8, 9, 9, 7, 5]

def mode (l : List ℕ) : Set ℕ := sorry

def variance (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def percentile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem scores_analysis :
  (mode scores = {7, 9}) ∧
  (variance scores = 3) ∧
  (mean scores = 7) ∧
  (percentile scores (70/100) = 17/2) := by sorry

end scores_analysis_l1609_160930


namespace negative_two_b_cubed_l1609_160967

theorem negative_two_b_cubed (b : ℝ) : (-2 * b)^3 = -8 * b^3 := by
  sorry

end negative_two_b_cubed_l1609_160967


namespace abc_product_l1609_160932

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : 1 / a + 1 / b + 1 / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := by
  sorry

end abc_product_l1609_160932


namespace unique_prime_solution_l1609_160970

theorem unique_prime_solution :
  ∃! (p q : ℕ) (n : ℕ), 
    Prime p ∧ Prime q ∧ n > 1 ∧
    (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) ∧
    p = 2 ∧ q = 5 ∧ n = 2 := by
  sorry

end unique_prime_solution_l1609_160970


namespace divisible_by_1998_digit_sum_l1609_160946

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end divisible_by_1998_digit_sum_l1609_160946


namespace optimal_promotional_expense_l1609_160957

noncomputable section

-- Define the sales volume function
def P (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def profit (x : ℝ) : ℝ := 16 - (4 / (x + 1) + x)

-- Define the theorem
theorem optimal_promotional_expense (a : ℝ) (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x ≤ profit (min 1 a)) ∧
  (a ≥ 1 → profit 1 = (profit ∘ min 1) a) ∧
  (a < 1 → profit a = (profit ∘ min 1) a) := by
  sorry

end

end optimal_promotional_expense_l1609_160957


namespace pine_boys_count_l1609_160989

/-- Represents a middle school in the winter program. -/
inductive School
| Maple
| Pine
| Oak

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the winter program. -/
structure WinterProgram where
  total_students : Nat
  total_boys : Nat
  total_girls : Nat
  maple_students : Nat
  pine_students : Nat
  oak_students : Nat
  maple_girls : Nat

/-- Theorem stating that the number of boys from Pine Middle School is 20. -/
theorem pine_boys_count (wp : WinterProgram) 
  (h1 : wp.total_students = 120)
  (h2 : wp.total_boys = 68)
  (h3 : wp.total_girls = 52)
  (h4 : wp.maple_students = 50)
  (h5 : wp.pine_students = 40)
  (h6 : wp.oak_students = 30)
  (h7 : wp.maple_girls = 22)
  (h8 : wp.total_students = wp.total_boys + wp.total_girls)
  (h9 : wp.total_students = wp.maple_students + wp.pine_students + wp.oak_students) :
  ∃ (pine_boys : Nat), pine_boys = 20 ∧ 
    pine_boys + (wp.pine_students - pine_boys) = wp.pine_students :=
  sorry


end pine_boys_count_l1609_160989


namespace correct_remaining_leaves_l1609_160940

/-- Calculates the number of remaining leaves on a tree at the end of summer --/
def remaining_leaves (branches : ℕ) (twigs_per_branch : ℕ) 
  (spring_3_leaf_percent : ℚ) (spring_4_leaf_percent : ℚ) (spring_5_leaf_percent : ℚ)
  (summer_leaf_increase : ℕ) (caterpillar_eaten_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the correct number of remaining leaves --/
theorem correct_remaining_leaves :
  remaining_leaves 100 150 (20/100) (30/100) (50/100) 2 (10/100) = 85050 :=
sorry

end correct_remaining_leaves_l1609_160940


namespace browns_house_number_l1609_160901

theorem browns_house_number :
  ∃! (n t : ℕ),
    20 < t ∧ t < 500 ∧
    1 ≤ n ∧ n ≤ t ∧
    n * (n + 1) = t * (t + 1) / 2 ∧
    n = 84 := by
  sorry

end browns_house_number_l1609_160901


namespace sum_of_a_and_b_l1609_160908

theorem sum_of_a_and_b (a b c d : ℝ) 
  (h1 : a * c + b * d + b * c + a * d = 48)
  (h2 : c + d = 8) : 
  a + b = 6 := by
sorry

end sum_of_a_and_b_l1609_160908


namespace equation_solution_l1609_160988

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (x / (x - 1) - 1 = 1) :=
by
  use 2
  constructor
  · constructor
    · norm_num
    · field_simp
      ring
  · intro y hy
    have h1 : y ≠ 1 := hy.1
    have h2 : y / (y - 1) - 1 = 1 := hy.2
    -- Proof steps would go here
    sorry

#check equation_solution

end equation_solution_l1609_160988


namespace simplify_fraction_l1609_160976

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end simplify_fraction_l1609_160976


namespace direction_vector_valid_l1609_160909

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Converts a parametric equation to a line -/
def parametricToLine (x0 x1 y0 y1 : ℝ) : Line2D :=
  { a := y1 - y0, b := x0 - x1, c := x0 * y1 - x1 * y0 }

/-- Checks if a vector is parallel to a line -/
def isParallel (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

/-- The given parametric equation of line l -/
def lineL : Line2D :=
  parametricToLine 1 3 2 1

/-- The proposed direction vector -/
def directionVector : Vector2D :=
  { x := -2, y := 1 }

theorem direction_vector_valid :
  isParallel directionVector lineL := by sorry

end direction_vector_valid_l1609_160909


namespace systematic_sampling_survey_c_count_l1609_160933

theorem systematic_sampling_survey_c_count 
  (total_population : Nat) 
  (sample_size : Nat) 
  (first_number : Nat) 
  (survey_c_lower_bound : Nat) 
  (survey_c_upper_bound : Nat) 
  (h1 : total_population = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 8)
  (h4 : survey_c_lower_bound = 751)
  (h5 : survey_c_upper_bound = 1000) :
  (Finset.filter (fun n => 
    let term := first_number + (n - 1) * (total_population / sample_size)
    term ≥ survey_c_lower_bound ∧ term ≤ survey_c_upper_bound
  ) (Finset.range sample_size)).card = 12 := by
  sorry

#check systematic_sampling_survey_c_count

end systematic_sampling_survey_c_count_l1609_160933


namespace function_bounds_bounds_achievable_l1609_160980

theorem function_bounds (x : ℝ) : 
  6 ≤ 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ∧ 
  7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ≤ 10 :=
by sorry

theorem bounds_achievable : 
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 6) ∧
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 10) :=
by sorry

end function_bounds_bounds_achievable_l1609_160980


namespace absolute_value_nonnegative_l1609_160982

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end absolute_value_nonnegative_l1609_160982


namespace root_sum_magnitude_l1609_160977

theorem root_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 9 = 0 →
  r₂^2 + p*r₂ + 9 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end root_sum_magnitude_l1609_160977


namespace smallest_n_squared_plus_n_divisibility_l1609_160921

theorem smallest_n_squared_plus_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k = 0)) ∧
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k ≠ 0)) ∧
  (∀ (m : ℕ), (m > 0) ∧ (m < n) → 
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k = 0)) ∨
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k ≠ 0))) ∧
  n = 3 :=
by sorry

end smallest_n_squared_plus_n_divisibility_l1609_160921


namespace tangent_line_to_sine_curve_l1609_160971

theorem tangent_line_to_sine_curve (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin (t + Real.pi / 3)
  let point : ℝ × ℝ := (0, Real.sqrt 3 / 2)
  let tangent_equation : ℝ → ℝ → Prop := λ x y => x - 2 * y + Real.sqrt 3 = 0
  (∀ t, f t = Real.sin (t + Real.pi / 3)) →
  (point.1 = 0 ∧ point.2 = Real.sqrt 3 / 2) →
  (∃ k, ∀ x, tangent_equation x (k * x + point.2)) →
  tangent_equation x y = (x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry


end tangent_line_to_sine_curve_l1609_160971


namespace water_flow_restrictor_l1609_160927

theorem water_flow_restrictor (original_rate : ℝ) (reduced_rate : ℝ) : 
  original_rate = 5 →
  reduced_rate = 0.6 * original_rate - 1 →
  reduced_rate = 2 := by
sorry

end water_flow_restrictor_l1609_160927


namespace swimming_area_probability_l1609_160913

theorem swimming_area_probability (lake_radius swimming_area_radius : ℝ) 
  (lake_radius_pos : 0 < lake_radius)
  (swimming_area_radius_pos : 0 < swimming_area_radius)
  (swimming_area_in_lake : swimming_area_radius ≤ lake_radius) :
  lake_radius = 5 → swimming_area_radius = 3 →
  (π * swimming_area_radius^2) / (π * lake_radius^2) = 9 / 25 := by
sorry

end swimming_area_probability_l1609_160913


namespace smallest_x_value_l1609_160936

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) : 
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (3 : ℚ) / 4 = w / (240 + z) :=
by sorry

#check smallest_x_value

end smallest_x_value_l1609_160936


namespace x_value_l1609_160937

theorem x_value (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 80 → x = 20 / 3 := by
  sorry

end x_value_l1609_160937


namespace sum_and_ratio_to_difference_l1609_160942

theorem sum_and_ratio_to_difference (m n : ℝ) 
  (sum_eq : m + n = 490)
  (ratio_eq : m / n = 1.2) : 
  ∃ (diff : ℝ), abs (m - n - diff) < 0.5 ∧ diff = 45 :=
sorry

end sum_and_ratio_to_difference_l1609_160942


namespace polynomial_evaluation_l1609_160961

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2*x^4 + 3*x^3 - x^2 + 2*x + 5
  f (-2) = 5 := by
  sorry

end polynomial_evaluation_l1609_160961


namespace quadratic_solution_l1609_160929

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
sorry

end quadratic_solution_l1609_160929


namespace face_covers_are_squares_and_rectangles_l1609_160969

/-- A parallelogram covering a face of a unit cube -/
structure FaceCover where
  -- The parallelogram's area
  area : ℝ
  -- The parallelogram is a square
  is_square : Prop
  -- The parallelogram is a rectangle
  is_rectangle : Prop

/-- A cube with edge length 1 covered by six identical parallelograms -/
structure CoveredCube where
  -- The edge length of the cube
  edge_length : ℝ
  -- The six identical parallelograms covering the cube
  face_covers : Fin 6 → FaceCover
  -- All face covers are identical
  covers_identical : ∀ (i j : Fin 6), face_covers i = face_covers j
  -- The edge length is 1
  edge_is_unit : edge_length = 1
  -- Each face cover has an area of 1
  cover_area_is_unit : ∀ (i : Fin 6), (face_covers i).area = 1

/-- Theorem: All face covers of a unit cube are squares and rectangles -/
theorem face_covers_are_squares_and_rectangles (cube : CoveredCube) :
  (∀ (i : Fin 6), (cube.face_covers i).is_square) ∧
  (∀ (i : Fin 6), (cube.face_covers i).is_rectangle) := by
  sorry


end face_covers_are_squares_and_rectangles_l1609_160969


namespace functional_equation_solution_l1609_160922

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of being twice differentiable with continuous second derivative
def TwiceDifferentiableContinuous (f : RealFunction) : Prop :=
  Differentiable ℝ f ∧ 
  Differentiable ℝ (deriv f) ∧ 
  Continuous (deriv (deriv f))

-- Define the functional equation
def SatisfiesFunctionalEquation (f : RealFunction) : Prop :=
  ∀ t : ℝ, f t ^ 2 = f (t * Real.sqrt 2)

-- Main theorem
theorem functional_equation_solution 
  (f : RealFunction) 
  (h1 : TwiceDifferentiableContinuous f) 
  (h2 : SatisfiesFunctionalEquation f) : 
  (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x^2)) ∨ 
  (∀ x : ℝ, f x = 0) := by
  sorry

end functional_equation_solution_l1609_160922


namespace sum_of_squares_positive_and_negative_l1609_160924

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_positive_and_negative :
  2 * (sum_of_squares 50) = 85850 := by sorry

end sum_of_squares_positive_and_negative_l1609_160924


namespace student_count_l1609_160968

/-- The number of students in the class -/
def n : ℕ := sorry

/-- The total number of tokens -/
def total_tokens : ℕ := 960

/-- The number of tokens each student gives to the teacher -/
def tokens_to_teacher : ℕ := 4

theorem student_count :
  (n > 0) ∧
  (total_tokens % n = 0) ∧
  (∃ k : ℕ, k > 0 ∧ total_tokens / n - tokens_to_teacher = k ∧ k * (n + 1) = total_tokens) →
  n = 15 := by sorry

end student_count_l1609_160968


namespace smallest_abs_rational_l1609_160985

theorem smallest_abs_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end smallest_abs_rational_l1609_160985


namespace symmetric_points_sum_l1609_160919

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨-2, 3⟩
  let q : Point := ⟨a, b⟩
  symmetricYAxis p q → a + b = 5 := by
  sorry

end symmetric_points_sum_l1609_160919


namespace problem_1_l1609_160973

theorem problem_1 : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end problem_1_l1609_160973


namespace fourth_grade_students_end_of_year_l1609_160956

/-- Calculates the total number of students at the end of the year given the initial number,
    students added during the year, and new students who came to school. -/
def total_students (initial : ℝ) (added : ℝ) (new_students : ℝ) : ℝ :=
  initial + added + new_students

/-- Proves that given the specific numbers in the problem, the total number of students
    at the end of the year is 56.0. -/
theorem fourth_grade_students_end_of_year :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end fourth_grade_students_end_of_year_l1609_160956


namespace jerome_money_ratio_l1609_160907

def jerome_money_problem (initial_money : ℕ) : Prop :=
  let meg_money : ℕ := 8
  let bianca_money : ℕ := 3 * meg_money
  let remaining_money : ℕ := 54
  initial_money = remaining_money + meg_money + bianca_money ∧
  (initial_money : ℚ) / remaining_money = 43 / 27

theorem jerome_money_ratio :
  ∃ (initial_money : ℕ), jerome_money_problem initial_money :=
sorry

end jerome_money_ratio_l1609_160907


namespace roses_planted_is_difference_l1609_160923

/-- The number of rose bushes planted in a park --/
def rosesBushesPlanted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of rose bushes planted is the difference between final and initial counts --/
theorem roses_planted_is_difference (initial final : ℕ) (h : final ≥ initial) :
  rosesBushesPlanted initial final = final - initial :=
by
  sorry

/-- Specific instance for the given problem --/
example : rosesBushesPlanted 2 6 = 4 :=
by
  sorry

end roses_planted_is_difference_l1609_160923


namespace two_digit_number_square_difference_l1609_160981

theorem two_digit_number_square_difference (a b : ℤ) 
  (h1 : a > b) (h2 : a + b = 10) : 
  ∃ k : ℤ, (9*a + 10)^2 - (100 - 9*a)^2 = 20 * k := by
  sorry

end two_digit_number_square_difference_l1609_160981


namespace geometric_sequence_problem_l1609_160984

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  (a 5) ^ 2 + 2016 * (a 5) + 9 = 0 →  -- a_5 is a root of the equation
  (a 9) ^ 2 + 2016 * (a 9) + 9 = 0 →  -- a_9 is a root of the equation
  a 7 = -3 := by
sorry

end geometric_sequence_problem_l1609_160984


namespace paper_cup_probability_l1609_160953

theorem paper_cup_probability (total_tosses : ℕ) (mouth_up_occurrences : ℕ) 
  (h1 : total_tosses = 200) (h2 : mouth_up_occurrences = 48) :
  (mouth_up_occurrences : ℚ) / total_tosses = 24 / 100 := by
  sorry

end paper_cup_probability_l1609_160953


namespace lowest_sale_price_percentage_l1609_160900

theorem lowest_sale_price_percentage (list_price : ℝ) (max_discount : ℝ) (additional_discount : ℝ) :
  list_price = 80 →
  max_discount = 0.5 →
  additional_discount = 0.2 →
  let discounted_price := list_price * (1 - max_discount)
  let final_price := discounted_price - (list_price * additional_discount)
  final_price / list_price = 0.3 := by sorry

end lowest_sale_price_percentage_l1609_160900


namespace shift_proportional_function_l1609_160915

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function vertically by a given amount -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

theorem shift_proportional_function :
  let f : LinearFunction := { m := -2, b := 0 }
  let shifted_f := verticalShift f 3
  shifted_f = { m := -2, b := 3 } := by
  sorry

end shift_proportional_function_l1609_160915


namespace art_gallery_visitors_prove_initial_girls_l1609_160949

theorem art_gallery_visitors : ℕ → ℕ → Prop :=
  fun girls boys =>
    -- After 15 girls left, there were twice as many boys as girls remaining
    boys = 2 * (girls - 15) ∧
    -- After 45 boys left, there were five times as many girls as boys remaining
    (girls - 15) = 5 * (boys - 45) ∧
    -- The number of girls initially in the gallery is 40
    girls = 40

-- The theorem to prove
theorem prove_initial_girls : ∃ (girls boys : ℕ), art_gallery_visitors girls boys :=
  sorry

end art_gallery_visitors_prove_initial_girls_l1609_160949


namespace number_exceeding_half_by_80_l1609_160959

theorem number_exceeding_half_by_80 (x : ℝ) : x = 0.5 * x + 80 → x = 160 := by
  sorry

end number_exceeding_half_by_80_l1609_160959


namespace shortest_distance_ln_to_line_l1609_160917

/-- The shortest distance from a point on the curve y = ln x to the line y = x + 1 is √2 -/
theorem shortest_distance_ln_to_line : 
  ∃ (x y : ℝ), y = Real.log x ∧ 
  (∀ (x' y' : ℝ), y' = Real.log x' → 
    Real.sqrt 2 ≤ Real.sqrt ((x' - x)^2 + (y' - (x + 1))^2)) := by
  sorry

end shortest_distance_ln_to_line_l1609_160917


namespace geometric_sum_of_root_l1609_160950

theorem geometric_sum_of_root (x : ℝ) : 
  x^10 - 3*x + 2 = 0 → x ≠ 1 → x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end geometric_sum_of_root_l1609_160950


namespace line_perpendicular_to_plane_l1609_160925

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) :
  parallel m n → 
  perpendicular_line_plane n β → 
  perpendicular_line_plane m β :=
sorry

end line_perpendicular_to_plane_l1609_160925


namespace heejin_has_most_volleyballs_l1609_160962

/-- The number of basketballs Heejin has -/
def basketballs : ℕ := 3

/-- The number of volleyballs Heejin has -/
def volleyballs : ℕ := 5

/-- The number of baseballs Heejin has -/
def baseball : ℕ := 1

/-- Theorem stating that Heejin has more volleyballs than any other type of ball -/
theorem heejin_has_most_volleyballs : 
  volleyballs > basketballs ∧ volleyballs > baseball :=
sorry

end heejin_has_most_volleyballs_l1609_160962


namespace defective_clock_correct_time_fraction_l1609_160974

/-- Represents a 12-hour digital clock with a defect that displays 1 instead of 2 --/
structure DefectiveClock :=
  (hours : Fin 12)
  (minutes : Fin 60)

/-- Checks if the given hour is displayed correctly --/
def hour_correct (h : Fin 12) : Bool :=
  h ≠ 2 ∧ h ≠ 12

/-- Checks if the given minute is displayed correctly --/
def minute_correct (m : Fin 60) : Bool :=
  m % 10 ≠ 2 ∧ m / 10 ≠ 2

/-- The fraction of the day during which the clock displays the correct time --/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (5 : ℚ) / 8

theorem defective_clock_correct_time_fraction :
  ∀ (clock : DefectiveClock),
  correct_time_fraction clock = (5 : ℚ) / 8 :=
by sorry

end defective_clock_correct_time_fraction_l1609_160974


namespace train_length_l1609_160993

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 18 → speed_kmh * (1000 / 3600) * time_s = 900 := by
  sorry

#check train_length

end train_length_l1609_160993


namespace tempo_value_calculation_l1609_160972

/-- The original value of a tempo given insurance and premium information -/
def tempoOriginalValue (insuredFraction : ℚ) (premiumRate : ℚ) (premiumAmount : ℚ) : ℚ :=
  premiumAmount / (premiumRate * insuredFraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_value_calculation :
  let insuredFraction : ℚ := 4 / 5
  let premiumRate : ℚ := 13 / 1000
  let premiumAmount : ℚ := 910
  tempoOriginalValue insuredFraction premiumRate premiumAmount = 87500 := by
  sorry

#eval tempoOriginalValue (4/5) (13/1000) 910

end tempo_value_calculation_l1609_160972


namespace algebraic_expression_inconsistency_l1609_160954

theorem algebraic_expression_inconsistency (a b : ℤ) :
  (-a + b = -1) ∧ (a + b = 5) ∧ (4*a + b = 14) →
  (2*a + b ≠ 7) :=
by sorry

end algebraic_expression_inconsistency_l1609_160954


namespace ellipse_parameter_sum_l1609_160943

-- Define the foci
def F₁ : ℝ × ℝ := (0, 4)
def F₂ : ℝ × ℝ := (6, 4)

-- Define the ellipse
def Ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  Real.sqrt ((x - x₁)^2 + (y - y₁)^2) + Real.sqrt ((x - x₂)^2 + (y - y₂)^2) = 10

-- Define the ellipse equation parameters
def h : ℝ := sorry
def k : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem ellipse_parameter_sum :
  h + k + a + b = 16 := by sorry

end ellipse_parameter_sum_l1609_160943


namespace gcd_factorial_ratio_l1609_160952

theorem gcd_factorial_ratio : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 480 := by
  sorry

end gcd_factorial_ratio_l1609_160952


namespace circle_radius_tangent_to_lines_l1609_160983

/-- Given a circle with center (0,k) where k > 6, if the circle is tangent to the lines y = x, y = -x, and y = 6, then its radius is 6√2 + 6. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 6) :
  let C := { p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - k)^2 = r^2 }
  let L1 := { p : ℝ × ℝ | p.2 = p.1 }
  let L2 := { p : ℝ × ℝ | p.2 = -p.1 }
  let L3 := { p : ℝ × ℝ | p.2 = 6 }
  (∃ (p1 : ℝ × ℝ), p1 ∈ C ∧ p1 ∈ L1) →
  (∃ (p2 : ℝ × ℝ), p2 ∈ C ∧ p2 ∈ L2) →
  (∃ (p3 : ℝ × ℝ), p3 ∈ C ∧ p3 ∈ L3) →
  r = 6 * (Real.sqrt 2 + 1) :=
by
  sorry

end circle_radius_tangent_to_lines_l1609_160983


namespace factors_of_12650_l1609_160903

theorem factors_of_12650 : Nat.card (Nat.divisors 12650) = 24 := by
  sorry

end factors_of_12650_l1609_160903


namespace car_speed_problem_l1609_160997

/-- Proves that car R's speed is 30 mph given the conditions of the problem -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ)
  (h1 : distance = 300)
  (h2 : time_diff = 2)
  (h3 : speed_diff = 10)
  (h4 : distance / (car_r_speed + speed_diff) + time_diff = distance / car_r_speed)
  : car_r_speed = 30 :=
by
  sorry

#check car_speed_problem

end car_speed_problem_l1609_160997


namespace subtract_negative_four_minus_negative_seven_l1609_160979

theorem subtract_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem four_minus_negative_seven : 4 - (-7) = 11 := by sorry

end subtract_negative_four_minus_negative_seven_l1609_160979


namespace angle_identities_l1609_160945

theorem angle_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.cos α = 1 / 3) : 
  Real.tan α = 2 * Real.sqrt 2 ∧ 
  (Real.sqrt 2 * Real.sin (Real.pi + α) + 2 * Real.cos α) / 
  (Real.cos α - Real.sqrt 2 * Real.cos (Real.pi / 2 + α)) = -2 / 5 := by
sorry

end angle_identities_l1609_160945


namespace compound_interest_theorem_specific_case_calculation_l1609_160986

/-- Compound interest calculation function -/
def compound_interest (a : ℝ) (r : ℝ) (x : ℕ) : ℝ :=
  a * (1 + r) ^ x

/-- Theorem for compound interest calculation -/
theorem compound_interest_theorem (a r : ℝ) (x : ℕ) :
  compound_interest a r x = a * (1 + r) ^ x :=
by sorry

/-- Specific case calculation -/
theorem specific_case_calculation :
  let a : ℝ := 1000
  let r : ℝ := 0.0225
  let x : ℕ := 4
  abs (compound_interest a r x - 1093.08) < 0.01 :=
by sorry

end compound_interest_theorem_specific_case_calculation_l1609_160986


namespace circles_intersect_l1609_160926

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 - 12*x + y^2 - 8*y - 12 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 - 10*y + 34 = 0

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := 0

/-- Theorem stating that the shortest distance between the two circles is 0 -/
theorem circles_intersect : 
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ shortest_distance = 0 :=
sorry

end circles_intersect_l1609_160926


namespace number_fraction_proof_l1609_160941

theorem number_fraction_proof (N : ℝ) (h : (3/10) * N - 8 = 12) : (1/5) * N = 40/3 := by
  sorry

end number_fraction_proof_l1609_160941


namespace gcd_119_34_l1609_160964

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end gcd_119_34_l1609_160964


namespace common_ratio_of_geometric_series_l1609_160958

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 7/8
  | 1 => -14/27
  | 2 => 56/81
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, r = -2/3 ∧
    ∀ n : ℕ, n > 0 → geometric_series n = geometric_series (n-1) * r :=
sorry

end common_ratio_of_geometric_series_l1609_160958


namespace correct_product_l1609_160963

-- Define a function to reverse the digits of a three-digit number
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

-- Define the theorem
theorem correct_product (a b : ℕ) : 
  (100 ≤ a ∧ a < 1000) →  -- a is a three-digit number
  (0 < b) →               -- b is positive
  (reverse_digits a * b = 396) →  -- erroneous product condition
  (a * b = 693) :=        -- correct product
by sorry

end correct_product_l1609_160963


namespace rectangle_formations_l1609_160918

theorem rectangle_formations (h : ℕ) (v : ℕ) (h_val : h = 5) (v_val : v = 4) :
  (Nat.choose h 2) * (Nat.choose v 2) = 60 :=
by sorry

end rectangle_formations_l1609_160918


namespace system_of_equations_solution_system_of_inequalities_solution_l1609_160916

-- Part 1: System of equations
theorem system_of_equations_solution :
  let x : ℚ := 10
  let y : ℚ := 8/3
  (x / 3 + y / 4 = 4) ∧ (2 * x - 3 * y = 12) := by sorry

-- Part 2: System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℚ, -1 ≤ x ∧ x < 3 →
    (x / 3 > (x - 1) / 2) ∧ (3 * (x + 2) ≥ 2 * x + 5) := by sorry

end system_of_equations_solution_system_of_inequalities_solution_l1609_160916


namespace least_positive_integer_congruence_l1609_160987

theorem least_positive_integer_congruence (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 9 = 8) ∧ 
  (∀ x : ℕ, x < b → ¬((x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ (x % 9 = 8))) →
  b = 179 :=
by sorry

end least_positive_integer_congruence_l1609_160987


namespace original_number_of_people_l1609_160938

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) - (x / 3 : ℚ) / 4 = 15 → x = 60 := by
  sorry

end original_number_of_people_l1609_160938


namespace scooter_price_l1609_160966

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price → 
  total_price = 1200 := by
sorry

end scooter_price_l1609_160966


namespace prism_height_l1609_160975

/-- Regular prism with base ABC and top A₁B₁C₁ -/
structure RegularPrism where
  a : ℝ  -- side length of the base
  h : ℝ  -- height of the prism
  M : ℝ × ℝ × ℝ  -- midpoint of AC
  N : ℝ × ℝ × ℝ  -- midpoint of A₁B₁

/-- The projection of MN onto BA₁ is a/(2√6) -/
def projection_condition (prism : RegularPrism) : Prop :=
  ∃ (proj : ℝ), proj = prism.a / (2 * Real.sqrt 6)

/-- The theorem stating the possible heights of the prism -/
theorem prism_height (prism : RegularPrism) 
  (h_proj : projection_condition prism) :
  prism.h = prism.a / Real.sqrt 2 ∨ 
  prism.h = prism.a / (2 * Real.sqrt 6) := by
  sorry

end prism_height_l1609_160975


namespace sum_of_coordinates_after_reflection_l1609_160995

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflect_over_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The sum of coordinate values of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let c : Point := ⟨x, 8⟩
  let d : Point := reflect_over_x_axis c
  sum_of_coordinates c d = 2 * x := by
  sorry

end sum_of_coordinates_after_reflection_l1609_160995


namespace find_k_l1609_160978

theorem find_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end find_k_l1609_160978


namespace field_division_l1609_160935

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 500) (h2 : smaller_area = 225) :
  ∃ (larger_area difference_value : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = difference_value / 5 ∧
    difference_value = 250 :=
by sorry

end field_division_l1609_160935


namespace milk_price_calculation_l1609_160948

/-- Calculates the price per gallon of milk given the daily production, 
    number of days, and total income. -/
def price_per_gallon (daily_production : ℕ) (days : ℕ) (total_income : ℚ) : ℚ :=
  total_income / (daily_production * days)

/-- Theorem stating that the price per gallon of milk is $3.05 given the conditions. -/
theorem milk_price_calculation : 
  price_per_gallon 200 30 18300 = 305/100 := by
  sorry

#eval price_per_gallon 200 30 18300

end milk_price_calculation_l1609_160948


namespace curve_and_tangent_lines_l1609_160906

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

-- Define point N
def N : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem curve_and_tangent_lines :
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 + 2*x - 3 = 0) ∧
  (∀ x y : ℝ, (C x y ∧ (x - N.1)^2 + (y - N.2)^2 = 0) →
    (x = 1 ∨ 5*x - 12*y + 31 = 0)) := by
  sorry

end curve_and_tangent_lines_l1609_160906


namespace savings_proof_l1609_160928

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the person's savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end savings_proof_l1609_160928


namespace dad_jayson_age_ratio_l1609_160911

/-- Represents the ages and relationships in Jayson's family -/
structure Family where
  jayson_age : ℕ
  mom_age : ℕ
  dad_age : ℕ
  mom_age_at_birth : ℕ

/-- The conditions given in the problem -/
def problem_conditions (f : Family) : Prop :=
  f.jayson_age = 10 ∧
  f.mom_age = f.mom_age_at_birth + f.jayson_age ∧
  f.dad_age = f.mom_age + 2 ∧
  f.mom_age_at_birth = 28

/-- The theorem stating the ratio of Jayson's dad's age to Jayson's age -/
theorem dad_jayson_age_ratio (f : Family) :
  problem_conditions f → (f.dad_age : ℚ) / f.jayson_age = 4 := by
  sorry

end dad_jayson_age_ratio_l1609_160911


namespace min_value_sqrt_expression_l1609_160998

theorem min_value_sqrt_expression (x : ℝ) :
  Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) ≥ Real.sqrt 7 ∧
  (Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) = Real.sqrt 7 ↔ x = Real.sqrt 3 / 4 ∨ x = -Real.sqrt 3 / 4) :=
by sorry

end min_value_sqrt_expression_l1609_160998


namespace division_of_fraction_by_integer_l1609_160902

theorem division_of_fraction_by_integer :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end division_of_fraction_by_integer_l1609_160902


namespace monica_second_third_classes_l1609_160996

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  total_classes : Nat
  first_class : Nat
  fourth_class : Nat
  fifth_sixth_classes : Nat
  total_students : Nat

/-- The number of students in Monica's second and third classes combined -/
def students_in_second_third_classes (m : MonicasClasses) : Nat :=
  m.total_students - (m.first_class + m.fourth_class + m.fifth_sixth_classes)

/-- Theorem stating the number of students in Monica's second and third classes -/
theorem monica_second_third_classes :
  ∀ (m : MonicasClasses),
  m.total_classes = 6 →
  m.first_class = 20 →
  m.fourth_class = m.first_class / 2 →
  m.fifth_sixth_classes = 28 * 2 →
  m.total_students = 136 →
  students_in_second_third_classes m = 50 := by
  sorry

end monica_second_third_classes_l1609_160996


namespace subtraction_of_fractions_l1609_160934

theorem subtraction_of_fractions : 
  (5 : ℚ) / 6 - 1 / 6 - 1 / 4 = 5 / 12 := by sorry

end subtraction_of_fractions_l1609_160934


namespace min_value_of_quadratic_l1609_160905

/-- The function f(x) = x^2 - 2x + 3 has a minimum value of 2 for positive x -/
theorem min_value_of_quadratic (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ y, y > 0 → x^2 - 2*x + 3 ≥ min_val := by
  sorry

end min_value_of_quadratic_l1609_160905


namespace gift_contribution_theorem_l1609_160920

theorem gift_contribution_theorem (n : ℕ) (min_contribution max_contribution total : ℝ) :
  n = 12 →
  min_contribution = 1 →
  max_contribution = 9 →
  (∀ person, person ∈ Finset.range n → min_contribution ≤ person) →
  (∀ person, person ∈ Finset.range n → person ≤ max_contribution) →
  total = (n - 1) * min_contribution + max_contribution →
  total = 20 := by
  sorry

end gift_contribution_theorem_l1609_160920


namespace binomial_expansion_equal_coefficients_l1609_160947

theorem binomial_expansion_equal_coefficients (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end binomial_expansion_equal_coefficients_l1609_160947


namespace ellipse_symmetric_point_range_l1609_160951

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- Definition of symmetry with respect to y = 2x -/
def symmetric_points (x₀ y₀ x₁ y₁ : ℝ) : Prop :=
  (y₀ - y₁) / (x₀ - x₁) = -1/2 ∧ (y₀ + y₁) / 2 = 2 * ((x₀ + x₁) / 2)

/-- The main theorem -/
theorem ellipse_symmetric_point_range :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  ellipse_C x₀ y₀ →
  symmetric_points x₀ y₀ x₁ y₁ →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
sorry

end ellipse_symmetric_point_range_l1609_160951


namespace reciprocal_of_negative_fraction_l1609_160955

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_negative_fraction :
  reciprocal (-5/4) = -4/5 := by sorry

end reciprocal_of_negative_fraction_l1609_160955
