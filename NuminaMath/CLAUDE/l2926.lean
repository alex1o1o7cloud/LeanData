import Mathlib

namespace chords_and_triangles_10_points_l2926_292673

/-- The number of chords formed by n points on a circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of triangles formed by n points on a circumference -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem about chords and triangles formed by 10 points on a circumference -/
theorem chords_and_triangles_10_points :
  num_chords 10 = 45 ∧ num_triangles 10 = 120 := by
  sorry

end chords_and_triangles_10_points_l2926_292673


namespace johns_car_efficiency_l2926_292613

/-- Calculates the miles per gallon (MPG) of John's car based on his weekly driving habits. -/
def johns_car_mpg (work_miles_one_way : ℕ) (work_days : ℕ) (leisure_miles : ℕ) (gas_used : ℕ) : ℚ :=
  let total_miles := 2 * work_miles_one_way * work_days + leisure_miles
  total_miles / gas_used

/-- Proves that John's car gets 30 miles per gallon based on his weekly driving habits. -/
theorem johns_car_efficiency :
  johns_car_mpg 20 5 40 8 = 30 := by
  sorry

end johns_car_efficiency_l2926_292613


namespace shell_collection_sum_l2926_292682

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def shell_sum : ℕ := 
  let a₁ : ℕ := 2  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of shells collected is 345 -/
theorem shell_collection_sum : shell_sum = 345 := by
  sorry

end shell_collection_sum_l2926_292682


namespace trip_speed_calculation_l2926_292645

theorem trip_speed_calculation (v : ℝ) : 
  v > 0 → -- Ensuring speed is positive
  (35 / v + 35 / 24 = 70 / 32) → -- Average speed equation
  v = 48 := by
sorry

end trip_speed_calculation_l2926_292645


namespace binomial_representation_existence_and_uniqueness_l2926_292681

theorem binomial_representation_existence_and_uniqueness 
  (t l : ℕ) : 
  ∃! (m : ℕ) (a : ℕ → ℕ), 
    m ≤ l ∧ 
    (∀ i ∈ Finset.range (l - m + 1), a (m + i) ≥ m + i) ∧
    (∀ i ∈ Finset.range (l - m), a (m + i + 1) > a (m + i)) ∧
    t = (Finset.range (l - m + 1)).sum (λ i => Nat.choose (a (m + i)) (m + i)) :=
by sorry

end binomial_representation_existence_and_uniqueness_l2926_292681


namespace polar_to_cartesian_circle_l2926_292606

/-- The curve defined by r = 8 tan(θ)cos(θ) in polar coordinates is a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∃ (x₀ y₀ R : ℝ), ∀ (θ : ℝ) (r : ℝ),
    r = 8 * Real.tan θ * Real.cos θ →
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = R^2 := by
  sorry

end polar_to_cartesian_circle_l2926_292606


namespace hot_dog_eating_contest_l2926_292640

theorem hot_dog_eating_contest (first_competitor second_competitor third_competitor : ℕ) :
  first_competitor = 12 →
  third_competitor = 18 →
  third_competitor = (3 * second_competitor) / 4 →
  second_competitor / first_competitor = 2 := by
  sorry

end hot_dog_eating_contest_l2926_292640


namespace mean_proportional_segment_l2926_292671

theorem mean_proportional_segment (a c : ℝ) (x : ℝ) 
  (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end mean_proportional_segment_l2926_292671


namespace average_student_headcount_theorem_l2926_292697

/-- Represents the student headcount for a specific academic year --/
structure StudentCount where
  year : String
  count : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (q : ℚ) : ℤ :=
  (q + 1/2).floor

theorem average_student_headcount_theorem 
  (headcounts : List StudentCount)
  (error_margin : ℕ)
  (h1 : headcounts.length = 3)
  (h2 : error_margin = 50)
  (h3 : ∀ sc ∈ headcounts, sc.count ≥ 10000 ∧ sc.count ≤ 12000) :
  roundToNearest (average (headcounts.map (λ sc ↦ sc.count))) = 10833 := by
sorry

end average_student_headcount_theorem_l2926_292697


namespace min_value_theorem_l2926_292614

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 / x + 4 / y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 4 := by
  sorry

end min_value_theorem_l2926_292614


namespace arithmetic_sqrt_16_l2926_292618

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_sqrt_16_l2926_292618


namespace team_selection_problem_l2926_292665

def num_players : ℕ := 6
def team_size : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

theorem team_selection_problem :
  ways_to_select num_players team_size - ways_to_select (num_players - 1) (team_size - 1) = 100 := by
  sorry

end team_selection_problem_l2926_292665


namespace rectangle_placement_l2926_292688

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (x y : ℝ), x ≤ c ∧ y ≤ d ∧ x * y = a * b) ↔ 
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 := by
sorry

end rectangle_placement_l2926_292688


namespace sqrt_x_plus_2y_is_plus_minus_one_l2926_292621

theorem sqrt_x_plus_2y_is_plus_minus_one (x y : ℝ) 
  (h : Real.sqrt (x - 2) + abs (2 * y + 1) = 0) : 
  Real.sqrt (x + 2 * y) = 1 ∨ Real.sqrt (x + 2 * y) = -1 := by
  sorry

end sqrt_x_plus_2y_is_plus_minus_one_l2926_292621


namespace marco_painting_fraction_l2926_292626

theorem marco_painting_fraction (marco_rate carla_rate : ℚ) : 
  marco_rate = 1 / 60 →
  marco_rate + carla_rate = 1 / 40 →
  marco_rate * 32 = 8 / 15 := by
  sorry

end marco_painting_fraction_l2926_292626


namespace inequality_implies_lower_bound_l2926_292623

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end inequality_implies_lower_bound_l2926_292623


namespace beyonce_album_songs_l2926_292641

theorem beyonce_album_songs (singles : ℕ) (albums : ℕ) (songs_per_album : ℕ) (total_songs : ℕ) : 
  singles = 5 → albums = 2 → songs_per_album = 15 → total_songs = 55 → 
  total_songs - (singles + albums * songs_per_album) = 20 := by
sorry

end beyonce_album_songs_l2926_292641


namespace at_least_one_greater_than_one_l2926_292659

theorem at_least_one_greater_than_one (a b c : ℝ) : 
  (a - 1) * (b - 1) * (c - 1) > 0 → (a > 1 ∨ b > 1 ∨ c > 1) := by
  sorry

end at_least_one_greater_than_one_l2926_292659


namespace triangle_theorem_l2926_292630

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A > Real.sin t.C)
  (h2 : t.a * t.c * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end triangle_theorem_l2926_292630


namespace right_triangle_acute_angle_l2926_292628

theorem right_triangle_acute_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α = 90 →           -- one angle is 90° (right angle)
  β = 20 →           -- given angle is 20°
  γ = 70 :=          -- prove that the other acute angle is 70°
by sorry

end right_triangle_acute_angle_l2926_292628


namespace magnitude_of_z_l2926_292620

/-- Given a complex number z satisfying (z-i)i = 2+i, prove that |z| = √5 -/
theorem magnitude_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_z_l2926_292620


namespace equation_solution_l2926_292622

theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 1) = 3 / (4 - x)) ∧ (x = 1) := by
  sorry

end equation_solution_l2926_292622


namespace inverse_function_sum_l2926_292679

/-- Given a function g and its inverse g⁻¹, prove that c + d = 3 * (2^(1/3)) -/
theorem inverse_function_sum (c d : ℝ) 
  (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (hg : ∀ x, g x = c * x + d)
  (hg_inv : ∀ x, g_inv x = d * x - 2 * c)
  (h_inverse : ∀ x, g (g_inv x) = x) :
  c + d = 3 * Real.rpow 2 (1/3) := by
sorry

end inverse_function_sum_l2926_292679


namespace hollow_sphere_weight_double_radius_l2926_292667

/-- The weight of a hollow sphere given its radius -/
noncomputable def sphereWeight (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

theorem hollow_sphere_weight_double_radius (r : ℝ) (h : r > 0) :
  sphereWeight r = 8 → sphereWeight (2 * r) = 32 := by
  sorry

end hollow_sphere_weight_double_radius_l2926_292667


namespace inequality_range_l2926_292627

theorem inequality_range : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 1| - |x + 1| ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 1| - |x + 1| ≤ b) → b ≥ 2) :=
by sorry

end inequality_range_l2926_292627


namespace simplify_fraction_l2926_292642

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by sorry

end simplify_fraction_l2926_292642


namespace arithmetic_sequence_common_difference_l2926_292678

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 30)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 25) :
  d = 1 :=
sorry

end arithmetic_sequence_common_difference_l2926_292678


namespace min_value_implies_a_l2926_292646

theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3)) →
  (∀ x y : ℝ, x ≥ 1 → x + y ≤ 3 → y ≥ a * (x - 3) → 2 * x + y ≥ 1) →
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧ 2 * x + y = 1) →
  a = 1 / 2 :=
by sorry

end min_value_implies_a_l2926_292646


namespace vehicle_speeds_l2926_292616

/-- Proves that given the conditions, the bus speed is 20 km/h and the car speed is 60 km/h. -/
theorem vehicle_speeds 
  (distance : ℝ) 
  (bus_delay : ℝ) 
  (car_arrival_delay : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : distance = 70) 
  (h2 : bus_delay = 3) 
  (h3 : car_arrival_delay = 2/3) 
  (h4 : speed_ratio = 3) : 
  ∃ (bus_speed car_speed : ℝ), 
    bus_speed = 20 ∧ 
    car_speed = 60 ∧ 
    distance / bus_speed = distance / car_speed + bus_delay - car_arrival_delay ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end vehicle_speeds_l2926_292616


namespace second_number_problem_l2926_292677

theorem second_number_problem (A B : ℝ) : 
  A = 580 → 0.20 * A = 0.30 * B + 80 → B = 120 := by
sorry

end second_number_problem_l2926_292677


namespace function_equation_solution_l2926_292662

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end function_equation_solution_l2926_292662


namespace jerry_added_eleven_action_figures_l2926_292672

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℤ :=
  final - initial + removed

/-- Proof that Jerry added 11 action figures to his shelf -/
theorem jerry_added_eleven_action_figures :
  action_figures_added 7 10 8 = 11 := by
  sorry

end jerry_added_eleven_action_figures_l2926_292672


namespace base3_20121_equals_178_l2926_292663

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_20121_equals_178 : 
  base3ToBase10 [2, 0, 1, 2, 1] = 178 := by
  sorry

end base3_20121_equals_178_l2926_292663


namespace ball_probabilities_l2926_292634

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black = p_yellow + 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end ball_probabilities_l2926_292634


namespace product_evaluation_l2926_292612

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l2926_292612


namespace complex_roots_quadratic_l2926_292664

theorem complex_roots_quadratic (p q : ℝ) : 
  (p + 3*I : ℂ) * (p + 3*I : ℂ) - (12 + 11*I : ℂ) * (p + 3*I : ℂ) + (9 + 63*I : ℂ) = 0 ∧
  (q + 6*I : ℂ) * (q + 6*I : ℂ) - (12 + 11*I : ℂ) * (q + 6*I : ℂ) + (9 + 63*I : ℂ) = 0 →
  p = 9 ∧ q = 3 := by
sorry

end complex_roots_quadratic_l2926_292664


namespace only_t_squared_valid_l2926_292607

-- Define a type for programming statements
inductive ProgramStatement
  | Input (var : String) (value : String)
  | Assignment (var : String) (expr : String)
  | Print (var : String) (value : String)

-- Define a function to check if a statement is valid
def isValidStatement : ProgramStatement → Bool
  | ProgramStatement.Input var value => false  -- INPUT x=3 is not valid
  | ProgramStatement.Assignment "T" "T*T" => true  -- T=T*T is valid
  | ProgramStatement.Assignment var1 var2 => false  -- A=B=2 is not valid
  | ProgramStatement.Print var value => false  -- PRINT A=4 is not valid

-- Theorem stating that only T=T*T is valid among the given statements
theorem only_t_squared_valid :
  (isValidStatement (ProgramStatement.Input "x" "3") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "A" "B=2") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "T" "T*T") = true) ∧
  (isValidStatement (ProgramStatement.Print "A" "4") = false) := by
  sorry


end only_t_squared_valid_l2926_292607


namespace complex_modulus_sqrt_5_l2926_292633

theorem complex_modulus_sqrt_5 (a b : ℝ) (z : ℂ) : 
  (a + Complex.I)^2 = b * Complex.I → z = a + b * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_sqrt_5_l2926_292633


namespace profit_difference_is_183_50_l2926_292654

-- Define the given quantities
def cat_food_packages : ℕ := 9
def dog_food_packages : ℕ := 7
def cans_per_cat_package : ℕ := 15
def cans_per_dog_package : ℕ := 8
def cost_per_cat_package : ℚ := 14
def cost_per_dog_package : ℚ := 10
def price_per_cat_can : ℚ := 2.5
def price_per_dog_can : ℚ := 1.75

-- Define the profit calculation function
def profit_difference : ℚ :=
  let cat_revenue := (cat_food_packages * cans_per_cat_package : ℚ) * price_per_cat_can
  let dog_revenue := (dog_food_packages * cans_per_dog_package : ℚ) * price_per_dog_can
  let cat_cost := (cat_food_packages : ℚ) * cost_per_cat_package
  let dog_cost := (dog_food_packages : ℚ) * cost_per_dog_package
  (cat_revenue - cat_cost) - (dog_revenue - dog_cost)

-- Theorem statement
theorem profit_difference_is_183_50 : profit_difference = 183.5 := by sorry

end profit_difference_is_183_50_l2926_292654


namespace tangent_line_equation_l2926_292636

/-- The equation of the tangent line to y = ln x + x^2 at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log t + t^2
  let f' : ℝ → ℝ := λ t => 1/t + 2*t
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, 1)
  3*x - y - 2 = 0 ↔ y - point.2 = slope * (x - point.1) :=
by sorry

end tangent_line_equation_l2926_292636


namespace product_148_152_l2926_292689

theorem product_148_152 : 148 * 152 = 22496 := by
  sorry

end product_148_152_l2926_292689


namespace sqrt_9025_squared_l2926_292638

theorem sqrt_9025_squared : (Real.sqrt 9025)^2 = 9025 := by
  sorry

end sqrt_9025_squared_l2926_292638


namespace girls_average_age_l2926_292649

/-- Proves that the average age of girls is 11 years given the school's statistics -/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 604 →
  boys_avg_age = 12 →
  school_avg_age = 47/4 →
  num_girls = 151 →
  (total_students * school_avg_age - (total_students - num_girls) * boys_avg_age) / num_girls = 11 := by
  sorry


end girls_average_age_l2926_292649


namespace cubic_root_property_l2926_292603

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 4 and -3 are roots of the equation, then (b+c)/a = -13 -/
theorem cubic_root_property (a b c d : ℝ) (ha : a ≠ 0) :
  (a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0) →
  (a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) →
  (b + c) / a = -13 := by
  sorry

end cubic_root_property_l2926_292603


namespace alpha_value_when_beta_is_36_l2926_292675

/-- Given that α² is inversely proportional to β, and α = 4 when β = 9,
    prove that α = ±2 when β = 36. -/
theorem alpha_value_when_beta_is_36
  (k : ℝ)  -- Constant of proportionality
  (h1 : ∀ α β : ℝ, α ^ 2 * β = k)  -- α² is inversely proportional to β
  (h2 : 4 ^ 2 * 9 = k)  -- α = 4 when β = 9
  : {α : ℝ | α ^ 2 * 36 = k} = {2, -2} := by
  sorry

end alpha_value_when_beta_is_36_l2926_292675


namespace agri_product_sales_model_l2926_292600

/-- Agricultural product sales model -/
structure AgriProduct where
  cost_price : ℝ
  sales_quantity : ℝ → ℝ
  max_price : ℝ

/-- Daily sales profit function -/
def daily_profit (p : AgriProduct) (x : ℝ) : ℝ :=
  x * (p.sales_quantity x) - p.cost_price * (p.sales_quantity x)

/-- Theorem stating the properties of the agricultural product sales model -/
theorem agri_product_sales_model (p : AgriProduct) 
  (h_cost : p.cost_price = 20)
  (h_quantity : ∀ x, p.sales_quantity x = -2 * x + 80)
  (h_max_price : p.max_price = 30) :
  (∀ x, daily_profit p x = -2 * x^2 + 120 * x - 1600) ∧
  (∃ x, x ≤ p.max_price ∧ daily_profit p x = 150 ∧ x = 25) :=
sorry

end agri_product_sales_model_l2926_292600


namespace probability_different_specialties_l2926_292648

def total_students : ℕ := 50
def art_students : ℕ := 15
def dance_students : ℕ := 35

theorem probability_different_specialties :
  let total_combinations := total_students.choose 2
  let different_specialty_combinations := art_students * dance_students
  (different_specialty_combinations : ℚ) / total_combinations = 3 / 7 :=
sorry

end probability_different_specialties_l2926_292648


namespace mani_pedi_regular_price_l2926_292684

/-- The regular price of a mani/pedi, given a 25% discount, 5 purchases, and $150 total spent. -/
theorem mani_pedi_regular_price :
  ∀ (regular_price : ℝ),
  (regular_price * 0.75 * 5 = 150) →
  regular_price = 40 := by
sorry

end mani_pedi_regular_price_l2926_292684


namespace half_square_identity_l2926_292683

theorem half_square_identity (a : ℤ) : (a + 1/2)^2 = a * (a + 1) + 1/4 := by
  sorry

end half_square_identity_l2926_292683


namespace digit_150_is_1_l2926_292680

/-- The decimal expansion of 5/31 -/
def decimal_expansion : ℚ := 5 / 31

/-- The length of the repeating part in the decimal expansion of 5/31 -/
def repetition_length : ℕ := 15

/-- The position we're interested in -/
def target_position : ℕ := 150

/-- The function that returns the nth digit after the decimal point in the decimal expansion of 5/31 -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem digit_150_is_1 : nth_digit target_position = 1 := by
  sorry

end digit_150_is_1_l2926_292680


namespace radiator_problem_l2926_292676

/-- Represents the fraction of original substance remaining after multiple replacements -/
def fractionRemaining (totalVolume : ℚ) (replacementVolume : ℚ) (numberOfReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / totalVolume) ^ numberOfReplacements

/-- The radiator problem -/
theorem radiator_problem :
  let totalVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numberOfReplacements : ℕ := 3
  fractionRemaining totalVolume replacementVolume numberOfReplacements = 64 / 125 := by
  sorry

end radiator_problem_l2926_292676


namespace min_n_for_integer_sqrt_l2926_292608

theorem min_n_for_integer_sqrt (n : ℕ+) : 
  (∃ k : ℕ, k^2 = 51 + n) → (∀ m : ℕ+, m < n → ¬∃ k : ℕ, k^2 = 51 + m) → n = 13 := by
  sorry

end min_n_for_integer_sqrt_l2926_292608


namespace large_bulb_cost_l2926_292610

def prove_large_bulb_cost (small_bulbs : ℕ) (large_bulbs : ℕ) (initial_amount : ℕ) (small_bulb_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  small_bulbs = 3 →
  large_bulbs = 1 →
  initial_amount = 60 →
  small_bulb_cost = 8 →
  remaining_amount = 24 →
  (initial_amount - remaining_amount - small_bulbs * small_bulb_cost) / large_bulbs = 12

theorem large_bulb_cost : prove_large_bulb_cost 3 1 60 8 24 := by
  sorry

end large_bulb_cost_l2926_292610


namespace no_solution_iff_a_in_range_l2926_292609

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, 7 * |x - 4*a| + |x - a^2| + 6*x - 3*a ≠ 0) ↔ a < -17 ∨ a > 0 := by
  sorry

end no_solution_iff_a_in_range_l2926_292609


namespace geometric_sequence_k_value_l2926_292637

/-- A geometric sequence with sum S_n = 3 * 2^n + k -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  k : ℝ
  sum_formula : ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

/-- The value of k in the geometric sequence sum formula -/
theorem geometric_sequence_k_value (seq : GeometricSequence) : seq.k = -3 := by
  sorry


end geometric_sequence_k_value_l2926_292637


namespace circle_graph_percentage_l2926_292653

theorem circle_graph_percentage (total_degrees : ℝ) (total_percentage : ℝ) 
  (manufacturing_degrees : ℝ) (manufacturing_percentage : ℝ) : 
  total_degrees = 360 →
  total_percentage = 100 →
  manufacturing_degrees = 108 →
  manufacturing_percentage / total_percentage = manufacturing_degrees / total_degrees →
  manufacturing_percentage = 30 := by
sorry

end circle_graph_percentage_l2926_292653


namespace kite_parabolas_sum_l2926_292668

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure KiteParabolas where
  /-- Coefficient of x^2 in the first parabola y = ax^2 - 3 -/
  a : ℝ
  /-- Coefficient of x^2 in the second parabola y = 5 - bx^2 -/
  b : ℝ
  /-- The four intersection points form a kite -/
  is_kite : Bool
  /-- The area of the kite formed by the intersection points -/
  kite_area : ℝ
  /-- The parabolas intersect the coordinate axes in exactly four points -/
  four_intersections : Bool

/-- Theorem stating that under the given conditions, a + b = 128/81 -/
theorem kite_parabolas_sum (k : KiteParabolas) 
  (h1 : k.is_kite = true) 
  (h2 : k.kite_area = 18) 
  (h3 : k.four_intersections = true) : 
  k.a + k.b = 128/81 := by
  sorry

end kite_parabolas_sum_l2926_292668


namespace calculation_proof_l2926_292624

theorem calculation_proof :
  (Real.sqrt 48 * Real.sqrt (1/2) + Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 6 + 2 * Real.sqrt 3) ∧
  ((Real.sqrt 5 + 1)^2 + (Real.sqrt 13 + 3) * (Real.sqrt 13 - 3) = 10 + 2 * Real.sqrt 5) :=
by sorry

end calculation_proof_l2926_292624


namespace strawberry_milk_probability_l2926_292625

theorem strawberry_milk_probability :
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/5  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 20412/78125 := by
  sorry

end strawberry_milk_probability_l2926_292625


namespace cab_journey_time_l2926_292669

/-- Given a cab walking at 5/6 of its usual speed and arriving 8 minutes late,
    prove that its usual time to cover the journey is 40 minutes. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (5 / 6 * usual_speed) * (usual_time + 8) = usual_speed * usual_time → 
  usual_time = 40 := by
  sorry

end cab_journey_time_l2926_292669


namespace train_speed_l2926_292692

/-- Proves that the current average speed of a train is 48 kmph given the specified conditions -/
theorem train_speed (distance : ℝ) : 
  (distance = (50 / 60) * 48) → 
  (distance = (40 / 60) * 60) → 
  48 = (60 * 40) / 50 := by
  sorry

#check train_speed

end train_speed_l2926_292692


namespace geometric_sequence_sum_l2926_292691

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 5 + a 6 + a 7 + a 8 = 80 := by
  sorry

end geometric_sequence_sum_l2926_292691


namespace expression_simplification_l2926_292647

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end expression_simplification_l2926_292647


namespace union_A_complement_B_equals_result_l2926_292657

-- Define the set I
def I : Set ℤ := {x | |x| < 3}

-- Define set A
def A : Set ℤ := {1, 2}

-- Define set B
def B : Set ℤ := {-2, -1, 2}

-- Theorem statement
theorem union_A_complement_B_equals_result : A ∪ (I \ B) = {0, 1, 2} := by
  sorry

end union_A_complement_B_equals_result_l2926_292657


namespace bubble_sort_correct_l2926_292699

def bubble_sort (xs : List Int) : List Int :=
  let rec pass (ys : List Int) : List Int :=
    match ys with
    | [] => []
    | [x] => [x]
    | x :: y :: rest =>
      if x > y
      then y :: pass (x :: rest)
      else x :: pass (y :: rest)
  let rec sort (zs : List Int) (n : Nat) : List Int :=
    if n = 0 then zs
    else sort (pass zs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubble_sort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end bubble_sort_correct_l2926_292699


namespace jims_estimate_l2926_292687

theorem jims_estimate (x y ε : ℝ) (hx : x > y) (hy : y > 0) (hε : ε > 0) :
  (x^2 + ε) - (y^2 - ε) > x^2 - y^2 := by
  sorry

end jims_estimate_l2926_292687


namespace total_students_surveyed_l2926_292617

theorem total_students_surveyed :
  let french_and_english : ℕ := 25
  let french_not_english : ℕ := 65
  let percent_not_french : ℚ := 55/100
  let total_students : ℕ := 200
  (french_and_english + french_not_english : ℚ) / total_students = 1 - percent_not_french :=
by sorry

end total_students_surveyed_l2926_292617


namespace test_modes_l2926_292670

/-- Represents the frequency of each score in the test --/
def score_frequency : List (Nat × Nat) := [
  (65, 2), (73, 1), (82, 1), (88, 1),
  (91, 1), (96, 4), (102, 1), (104, 4), (110, 3)
]

/-- Finds the modes of a list of score frequencies --/
def find_modes (frequencies : List (Nat × Nat)) : List Nat :=
  sorry

/-- States that 96 and 104 are the modes of the given score frequencies --/
theorem test_modes : find_modes score_frequency = [96, 104] := by
  sorry

end test_modes_l2926_292670


namespace parabola_directrix_l2926_292601

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 4x + 4) / 8, its directrix is y = -2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) : 
  p.a = 1/8 ∧ p.b = -1/2 ∧ p.c = 1/2 → d.y = -2 := by
  sorry


end parabola_directrix_l2926_292601


namespace yolanda_departure_time_yolanda_left_30_minutes_before_l2926_292661

/-- Prove that Yolanda left 30 minutes before her husband caught up to her. -/
theorem yolanda_departure_time 
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)  -- Convert 15 minutes to hours
  (h4 : catch_up_time = 15 / 60)  -- Convert 15 minutes to hours
  : yolanda_speed * (husband_delay + catch_up_time) = husband_speed * catch_up_time :=
by sorry

/-- Yolanda's departure time before being caught -/
def yolanda_departure_before_catch (yolanda_speed : ℝ) (husband_speed : ℝ) (husband_delay : ℝ) (catch_up_time : ℝ) : ℝ :=
  husband_delay + catch_up_time

/-- Prove that Yolanda left 30 minutes (0.5 hours) before her husband caught up to her -/
theorem yolanda_left_30_minutes_before
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)
  (h4 : catch_up_time = 15 / 60)
  : yolanda_departure_before_catch yolanda_speed husband_speed husband_delay catch_up_time = 0.5 :=
by sorry

end yolanda_departure_time_yolanda_left_30_minutes_before_l2926_292661


namespace digit_156_is_5_l2926_292651

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeating_block_length : ℕ := 6

/-- The position of the digit we're looking for -/
def target_position : ℕ := 156

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_156_is_5 : nth_digit (target_position - 1) = 5 := by sorry

end digit_156_is_5_l2926_292651


namespace shaded_area_between_tangent_circles_l2926_292666

theorem shaded_area_between_tangent_circles 
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : d = 4) :
  let area_shaded := π * r₂^2 - π * r₁^2
  area_shaded = 48 * π := by
sorry

end shaded_area_between_tangent_circles_l2926_292666


namespace maisy_earns_fifteen_more_l2926_292650

/-- Represents Maisy's job options and calculates the difference in earnings -/
def maisys_job_earnings_difference : ℝ :=
  let current_hours : ℝ := 8
  let current_wage : ℝ := 10
  let new_hours : ℝ := 4
  let new_wage : ℝ := 15
  let bonus : ℝ := 35
  let current_earnings := current_hours * current_wage
  let new_earnings := new_hours * new_wage + bonus
  new_earnings - current_earnings

/-- Theorem stating that Maisy will earn $15 more per week at her new job -/
theorem maisy_earns_fifteen_more : maisys_job_earnings_difference = 15 := by
  sorry

end maisy_earns_fifteen_more_l2926_292650


namespace problem_statement_l2926_292604

def x : ℕ := 18
def y : ℕ := 8
def z : ℕ := 2

theorem problem_statement :
  -- (A) The arithmetic mean of x and y is greater than their geometric mean
  (x + y) / 2 > Real.sqrt (x * y) ∧
  -- (B) The sum of x and z is greater than their product divided by the sum of x and y
  (x + z : ℝ) > (x * z : ℝ) / (x + y) ∧
  -- (C) If the product of x and z is fixed, their sum can be made arbitrarily large
  (∀ ε > 0, ∃ k > 0, k + (x * z : ℝ) / k > 1 / ε) ∧
  -- (D) The arithmetic mean of x, y, and z is NOT greater than the sum of their squares divided by their sum
  ¬((x + y + z : ℝ) / 3 > (x^2 + y^2 + z^2 : ℝ) / (x + y + z)) :=
by sorry

end problem_statement_l2926_292604


namespace initial_pens_count_l2926_292686

theorem initial_pens_count (P : ℕ) : P = 5 :=
  by
  have h1 : 2 * (P + 20) - 19 = 31 := by sorry
  sorry

end initial_pens_count_l2926_292686


namespace adjacent_angles_l2926_292615

theorem adjacent_angles (α β : ℝ) : 
  α + β = 180 →  -- sum of adjacent angles is 180°
  α = β + 30 →   -- one angle is 30° larger than the other
  (α = 105 ∧ β = 75) ∨ (α = 75 ∧ β = 105) := by
sorry

end adjacent_angles_l2926_292615


namespace binary_110011_is_51_l2926_292602

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_is_51_l2926_292602


namespace rectangular_field_area_l2926_292652

/-- The area of a rectangular field with given perimeter and width -/
theorem rectangular_field_area
  (perimeter : ℝ) (width : ℝ)
  (h_perimeter : perimeter = 70)
  (h_width : width = 15) :
  width * ((perimeter / 2) - width) = 300 :=
by sorry

end rectangular_field_area_l2926_292652


namespace triangle_properties_l2926_292698

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a^2 + t.b^2 - t.c^2) * Real.tan t.C = Real.sqrt 2 * t.a * t.b) :
  (t.C = π/4 ∨ t.C = 3*π/4) ∧ 
  (t.c = 2 ∧ t.b = 2 * Real.sqrt 2 → 
    1/2 * t.a * t.b * Real.sin t.C = 2) := by
  sorry


end triangle_properties_l2926_292698


namespace bookshop_unsold_percentage_l2926_292685

/-- The percentage of unsold books in a bookshop -/
def unsold_percentage (initial_stock : ℕ) (mon_sales tues_sales wed_sales thurs_sales fri_sales : ℕ) : ℚ :=
  (initial_stock - (mon_sales + tues_sales + wed_sales + thurs_sales + fri_sales)) / initial_stock * 100

/-- Theorem stating the percentage of unsold books for the given scenario -/
theorem bookshop_unsold_percentage :
  unsold_percentage 1300 75 50 64 78 135 = 69.15384615384615 := by
  sorry

end bookshop_unsold_percentage_l2926_292685


namespace largest_n_for_equation_l2926_292690

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  9^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14 ∧ 
  ∀ (n : ℕ+), n > 9 → ¬∃ (a b c : ℕ+), 
    n^2 = 2*a^2 + 2*b^2 + 2*c^2 + 4*a*b + 4*b*c + 4*c*a + 6*a + 6*b + 6*c - 14 :=
by sorry

end largest_n_for_equation_l2926_292690


namespace line_intersects_circle_l2926_292643

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (x₀ y₀ x y : ℝ) : Prop := x₀ * x + y₀ * y = 4

def PointOutsideCircle (x₀ y₀ : ℝ) : Prop := x₀^2 + y₀^2 > 4

theorem line_intersects_circle (x₀ y₀ : ℝ) 
  (h1 : PointOutsideCircle x₀ y₀) :
  ∃ x y : ℝ, Circle x y ∧ Line x₀ y₀ x y :=
sorry

end line_intersects_circle_l2926_292643


namespace tan_pi_fourth_equals_one_l2926_292629

theorem tan_pi_fourth_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end tan_pi_fourth_equals_one_l2926_292629


namespace potato_bundle_size_l2926_292639

theorem potato_bundle_size (total_potatoes : ℕ) (potato_bundle_price : ℚ)
  (total_carrots : ℕ) (carrots_per_bundle : ℕ) (carrot_bundle_price : ℚ)
  (total_revenue : ℚ) :
  total_potatoes = 250 →
  potato_bundle_price = 19/10 →
  total_carrots = 320 →
  carrots_per_bundle = 20 →
  carrot_bundle_price = 2 →
  total_revenue = 51 →
  ∃ (potatoes_per_bundle : ℕ),
    potatoes_per_bundle = 25 ∧
    (potato_bundle_price * (total_potatoes / potatoes_per_bundle : ℚ) +
     carrot_bundle_price * (total_carrots / carrots_per_bundle : ℚ) = total_revenue) :=
by sorry

end potato_bundle_size_l2926_292639


namespace stripe_area_on_cylindrical_silo_l2926_292632

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (height : ℝ) 
  (stripe_width : ℝ) 
  (h_diameter : diameter = 30) 
  (h_height : height = 80) 
  (h_stripe_width : stripe_width = 3) :
  stripe_width * height = 240 := by
  sorry

end stripe_area_on_cylindrical_silo_l2926_292632


namespace value_of_x_l2926_292605

theorem value_of_x : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 20) ∧ x = 820 := by
  sorry

end value_of_x_l2926_292605


namespace f_is_quadratic_l2926_292694

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x+3) = 0 -/
def f (x : ℝ) : ℝ := x * (x + 3)

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l2926_292694


namespace triangle_inequality_with_120_degree_angle_l2926_292696

/-- Given a triangle with sides a, b, and c, where an angle of 120 degrees lies opposite to side c,
    prove that a, c, and a + b satisfy the triangle inequality theorem. -/
theorem triangle_inequality_with_120_degree_angle 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (triangle_exists : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_120 : a^2 = b^2 + c^2 - b*c) : 
  a + c > a + b ∧ a + (a + b) > c ∧ c + (a + b) > a :=
by sorry

end triangle_inequality_with_120_degree_angle_l2926_292696


namespace spatial_relationships_l2926_292656

structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  intersects : Line → Plane → Prop
  parallel : Line → Line → Prop
  parallel_line_plane : Line → Plane → Prop
  perpendicular : Line → Line → Prop
  in_plane : Line → Plane → Prop

variable (S : Space3D)

theorem spatial_relationships :
  (∀ (a : S.Line) (α : S.Plane), S.intersects a α → ¬∃ (l : S.Line), S.in_plane l α ∧ S.parallel l a) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel_line_plane b α ∧ S.perpendicular a b ∧ S.parallel_line_plane a α) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel a b ∧ S.in_plane b α ∧ ¬S.parallel_line_plane a α) :=
by sorry

end spatial_relationships_l2926_292656


namespace isabellas_haircut_l2926_292660

/-- Isabella's haircut problem -/
theorem isabellas_haircut (original_length cut_length : ℕ) (h1 : original_length = 18) (h2 : cut_length = 9) :
  original_length - cut_length = 9 := by
  sorry

end isabellas_haircut_l2926_292660


namespace alternating_sum_coefficients_l2926_292635

theorem alternating_sum_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^2 * (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -31 :=
by sorry

end alternating_sum_coefficients_l2926_292635


namespace tangent_sum_difference_l2926_292695

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
sorry

end tangent_sum_difference_l2926_292695


namespace sunzi_carriage_problem_l2926_292655

theorem sunzi_carriage_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x / 3 = y + 2 ∧ x / 2 + 9 = y) ↔ (x / 3 = y - 2 ∧ (x - 9) / 2 = y) :=
by sorry

end sunzi_carriage_problem_l2926_292655


namespace product_of_square_roots_l2926_292644

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (7 * q^3) * Real.sqrt (8 * q^5) = 210 * q^4 * Real.sqrt q :=
by sorry

end product_of_square_roots_l2926_292644


namespace movie_count_theorem_l2926_292693

/-- The number of movies Timothy and Theresa watched in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010

theorem movie_count_theorem :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2010 = timothy_2009 + 7 →
    timothy_2009 = 24 →
    theresa_2010 = 2 * timothy_2010 →
    theresa_2009 = timothy_2009 / 2 →
    total_movies timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 129 :=
by
  sorry

end movie_count_theorem_l2926_292693


namespace problem_statement_l2926_292611

theorem problem_statement (θ : ℝ) 
  (h : Real.sin (π / 4 - θ) + Real.cos (π / 4 - θ) = 1 / 5) :
  Real.cos (2 * θ) = -24 / 25 := by
  sorry

end problem_statement_l2926_292611


namespace no_intersection_l2926_292631

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no intersection points
theorem no_intersection :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y :=
sorry

end no_intersection_l2926_292631


namespace geometric_sequence_product_l2926_292658

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 1 * a 9 = 16 → a 2 * a 5 * a 8 = 64 := by
  sorry

end geometric_sequence_product_l2926_292658


namespace price_increase_percentage_l2926_292674

theorem price_increase_percentage (new_price : ℝ) (h1 : new_price - 0.8 * new_price = 4) : 
  (new_price - (0.8 * new_price)) / (0.8 * new_price) = 0.25 := by
  sorry

end price_increase_percentage_l2926_292674


namespace star_sqrt_eleven_l2926_292619

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem star_sqrt_eleven : star (Real.sqrt 11) (Real.sqrt 11) = 44 := by
  sorry

end star_sqrt_eleven_l2926_292619
