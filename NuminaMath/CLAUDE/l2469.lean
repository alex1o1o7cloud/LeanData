import Mathlib

namespace least_valid_integer_l2469_246948

def is_valid (a : ℕ) : Prop :=
  a % 2 = 0 ∧ a % 3 = 1 ∧ a % 4 = 2

theorem least_valid_integer : ∃ (a : ℕ), is_valid a ∧ ∀ (b : ℕ), b < a → ¬(is_valid b) :=
by
  use 10
  sorry

end least_valid_integer_l2469_246948


namespace rank_difference_bound_l2469_246966

variable (n : ℕ) 
variable (hn : n ≥ 2)

theorem rank_difference_bound 
  (X Y : Matrix (Fin n) (Fin n) ℂ) : 
  Matrix.rank (X * Y) - Matrix.rank (Y * X) ≤ n / 2 := by
  sorry

end rank_difference_bound_l2469_246966


namespace internal_tangent_length_l2469_246933

theorem internal_tangent_length (r₁ r₂ R : ℝ) (h₁ : r₁ = 19) (h₂ : r₂ = 32) (h₃ : R = 100) :
  let d := R - r₁ + R - r₂
  2 * (r₁ * r₂ / d) * Real.sqrt ((d / (r₁ + r₂))^2 - 1) = 140 :=
by sorry

end internal_tangent_length_l2469_246933


namespace complex_arithmetic_result_l2469_246919

theorem complex_arithmetic_result : 
  let z₁ : ℂ := 2 - 3*I
  let z₂ : ℂ := -1 + 5*I
  let z₃ : ℂ := 1 + I
  (z₁ + z₂) * z₃ = -1 + 3*I := by sorry

end complex_arithmetic_result_l2469_246919


namespace A_subset_B_l2469_246937

variable {X : Type*} -- Domain of functions f and g
variable (f g : X → ℝ) -- Real-valued functions f and g
variable (a : ℝ) -- Real number a

def A (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x| + |g x| < a}

def B (f g : X → ℝ) (a : ℝ) : Set X :=
  {x : X | |f x + g x| < a}

theorem A_subset_B (h : a > 0) : A f g a ⊆ B f g a := by
  sorry

end A_subset_B_l2469_246937


namespace exists_m_divisible_by_2005_l2469_246949

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 :
  ∃ m : ℕ+, (3^100 * m.val + (3^100 - 1)) % 2005 = 0 := by
  sorry

end exists_m_divisible_by_2005_l2469_246949


namespace triangle_side_difference_l2469_246994

theorem triangle_side_difference (y : ℕ) : 
  (y > 0 ∧ y + 7 > 9 ∧ y + 9 > 7 ∧ 7 + 9 > y) →
  (∃ (max min : ℕ), 
    (∀ z : ℕ, (z > 0 ∧ z + 7 > 9 ∧ z + 9 > 7 ∧ 7 + 9 > z) → z ≤ max ∧ z ≥ min) ∧
    max - min = 12) :=
by sorry

end triangle_side_difference_l2469_246994


namespace simplify_expression_l2469_246964

theorem simplify_expression (x y : ℝ) : 3*x + 4*x + 5*y + 2*y = 7*x + 7*y := by
  sorry

end simplify_expression_l2469_246964


namespace max_expenditure_max_expected_expenditure_l2469_246981

-- Define the linear regression model
def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

-- State the theorem
theorem max_expenditure (x : ℝ) (e : ℝ) :
  x = 10 →
  0.8 * x + 2 + e ≤ 10.5 :=
by
  sorry

-- Define the constraint on e
def e_constraint (e : ℝ) : Prop := abs e ≤ 0.5

-- State the main theorem
theorem max_expected_expenditure (x : ℝ) :
  x = 10 →
  ∀ e, e_constraint e →
  linear_regression x 0.8 2 e ≤ 10.5 :=
by
  sorry

end max_expenditure_max_expected_expenditure_l2469_246981


namespace debt_doubling_time_l2469_246952

def interest_rate : ℝ := 0.07

theorem debt_doubling_time : 
  ∀ t : ℕ, t < 10 → (1 + interest_rate) ^ t ≤ 2 ∧ 
  (1 + interest_rate) ^ 10 > 2 := by sorry

end debt_doubling_time_l2469_246952


namespace problem_1_l2469_246978

theorem problem_1 : (-20) + 3 - (-5) - 7 = -19 := by sorry

end problem_1_l2469_246978


namespace min_width_rectangle_l2469_246993

theorem min_width_rectangle (w : ℝ) : w > 0 →
  w * (w + 20) ≥ 150 →
  ∀ x > 0, x * (x + 20) ≥ 150 → w ≤ x →
  w = 10 := by
sorry

end min_width_rectangle_l2469_246993


namespace suraya_caleb_difference_l2469_246958

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := kayla_apples - 5

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- Theorem stating the difference between Suraya's and Caleb's apple count -/
theorem suraya_caleb_difference : suraya_apples - caleb_apples = 12 := by
  sorry

end suraya_caleb_difference_l2469_246958


namespace sqrt_point_zero_nine_equals_point_three_l2469_246988

theorem sqrt_point_zero_nine_equals_point_three :
  Real.sqrt 0.09 = 0.3 := by
  sorry

end sqrt_point_zero_nine_equals_point_three_l2469_246988


namespace brick_height_calculation_l2469_246926

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick length and width, and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_thickness : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℝ) :
  wall_length = 200 →
  wall_width = 300 →
  wall_thickness = 2 →
  brick_length = 25 →
  brick_width = 11 →
  num_bricks = 72.72727272727273 →
  (wall_length * wall_width * wall_thickness) / (brick_length * brick_width * num_bricks) = 6 :=
by sorry

end brick_height_calculation_l2469_246926


namespace distribute_two_four_x_minus_one_l2469_246995

theorem distribute_two_four_x_minus_one (x : ℝ) : 2 * (4 * x - 1) = 8 * x - 2 := by
  sorry

end distribute_two_four_x_minus_one_l2469_246995


namespace lace_cost_per_meter_l2469_246975

-- Define the lengths in centimeters
def cuff_length : ℝ := 50
def hem_length : ℝ := 300
def ruffle_length : ℝ := 20
def total_cost : ℝ := 36

-- Define the number of cuffs and ruffles
def num_cuffs : ℕ := 2
def num_ruffles : ℕ := 5

-- Define the conversion factor from cm to m
def cm_to_m : ℝ := 100

-- Theorem to prove
theorem lace_cost_per_meter :
  let total_length := num_cuffs * cuff_length + hem_length + (hem_length / 3) + num_ruffles * ruffle_length
  let total_length_m := total_length / cm_to_m
  total_cost / total_length_m = 6 := by
  sorry

end lace_cost_per_meter_l2469_246975


namespace sine_matrix_det_zero_l2469_246992

open Real Matrix

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![sin 1, sin 2, sin 3; 
                                       sin 4, sin 5, sin 6; 
                                       sin 7, sin 8, sin 9]
  det A = 0 := by
sorry

end sine_matrix_det_zero_l2469_246992


namespace running_time_calculation_l2469_246906

/-- Proves that given the conditions, the time taken to cover the same distance while running is [(a + 2b) × (c + d)] / (3a - b) hours. -/
theorem running_time_calculation 
  (a b c d : ℕ+) -- a, b, c, and d are positive integers
  (walking_speed : ℝ := a + 2*b) -- Walking speed = (a + 2b) kmph
  (walking_time : ℝ := c + d) -- Walking time = (c + d) hours
  (running_speed : ℝ := 3*a - b) -- Running speed = (3a - b) kmph
  (k : ℝ := 3) -- Conversion factor k = 3
  (h : k * walking_speed = running_speed) -- Assumption that k * walking_speed = running_speed
  : 
  (walking_speed * walking_time) / running_speed = (a + 2*b) * (c + d) / (3*a - b) := 
by
  sorry


end running_time_calculation_l2469_246906


namespace infinitely_many_with_1989_ones_l2469_246909

/-- Count the number of ones in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem stating that there are infinitely many positive integers
    with 1989 ones in their binary representation -/
theorem infinitely_many_with_1989_ones :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ countOnes m = 1989 := by sorry

end infinitely_many_with_1989_ones_l2469_246909


namespace periodic_decimal_to_fraction_l2469_246963

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + (3 * (2 / 99))) → (2 + (3 * (2 / 99)) = 68 / 33) := by
  sorry

end periodic_decimal_to_fraction_l2469_246963


namespace sum_of_arithmetic_sequence_l2469_246967

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + a 13 = 10) : 
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
sorry

end sum_of_arithmetic_sequence_l2469_246967


namespace batsman_average_theorem_l2469_246900

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

/-- Theorem: If a batsman's average increases by 2 after scoring 80 in the 17th innings,
    then the new average is 48 -/
theorem batsman_average_theorem (b : Batsman) :
  b.innings = 16 →
  newAverage b 80 = b.average + 2 →
  newAverage b 80 = 48 := by
  sorry

#check batsman_average_theorem

end batsman_average_theorem_l2469_246900


namespace greatest_integer_quadratic_inequality_l2469_246982

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 30 ≤ 0 ∧ 
  (∀ (m : ℤ), m^2 - 11*m + 30 ≤ 0 → m ≤ n) ∧
  n = 6 := by
  sorry

end greatest_integer_quadratic_inequality_l2469_246982


namespace field_trip_students_l2469_246904

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) :
  van_capacity = 7 →
  num_vans = 6 →
  num_adults = 9 →
  (van_capacity * num_vans - num_adults : ℕ) = 33 := by
  sorry

end field_trip_students_l2469_246904


namespace matrix_operation_proof_l2469_246935

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 8; -3, 0]
  2 • A + B = !![3, 14; -5, 8] := by
  sorry

end matrix_operation_proof_l2469_246935


namespace line_equation_proof_l2469_246960

/-- Proves that the equation of a line with a slope angle of 135° and a y-intercept of -1 is y = -x - 1 -/
theorem line_equation_proof (x y : ℝ) : 
  (∃ (k b : ℝ), k = Real.tan (135 * π / 180) ∧ b = -1 ∧ y = k * x + b) ↔ y = -x - 1 := by
  sorry

end line_equation_proof_l2469_246960


namespace expression_simplification_l2469_246979

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) (h3 : x ≠ -1) : 
  (2*x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2*x + 1)) - 2*x / (x + 1) = -2 / (x + 1) := by
  sorry

end expression_simplification_l2469_246979


namespace total_hours_worked_l2469_246977

theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) :
  hours_per_day = 3 →
  days_worked = 5 →
  total_hours = hours_per_day * days_worked →
  total_hours = 15 :=
by sorry

end total_hours_worked_l2469_246977


namespace mary_total_spending_l2469_246927

/-- The total amount Mary spent on clothing, given the costs of a shirt and a jacket. -/
def total_spent (shirt_cost jacket_cost : ℚ) : ℚ :=
  shirt_cost + jacket_cost

/-- Theorem stating that Mary's total spending is $25.31 -/
theorem mary_total_spending :
  total_spent 13.04 12.27 = 25.31 := by
  sorry

end mary_total_spending_l2469_246927


namespace min_students_is_minimum_l2469_246921

/-- The minimum number of students in the circle -/
def min_students : ℕ := 37

/-- Congcong's numbers are congruent modulo the number of students -/
axiom congcong_congruence : 25 ≡ 99 [ZMOD min_students]

/-- Mingming's numbers are congruent modulo the number of students -/
axiom mingming_congruence : 8 ≡ 119 [ZMOD min_students]

/-- The number of students is the minimum positive integer satisfying both congruences -/
theorem min_students_is_minimum :
  ∀ m : ℕ, m > 0 → (25 ≡ 99 [ZMOD m] ∧ 8 ≡ 119 [ZMOD m]) → m ≥ min_students :=
by sorry

end min_students_is_minimum_l2469_246921


namespace jack_sugar_today_l2469_246941

/-- The amount of sugar Jack has today -/
def S : ℕ := by sorry

/-- Theorem: Jack has 65 pounds of sugar today -/
theorem jack_sugar_today : S = 65 := by
  have h1 : S - 18 + 50 = 97 := by sorry
  sorry


end jack_sugar_today_l2469_246941


namespace car_speed_problem_l2469_246912

/-- Proves that given the conditions of the car problem, the average speed of Car X is 50 mph -/
theorem car_speed_problem (Vx : ℝ) : 
  (∃ (T : ℝ), 
    T > 0 ∧ 
    Vx * 1.2 + Vx * T = 50 * T ∧ 
    Vx * T = 98) → 
  Vx = 50 := by
sorry

end car_speed_problem_l2469_246912


namespace two_cars_meeting_l2469_246953

/-- Two cars meeting on a highway problem -/
theorem two_cars_meeting (highway_length : ℝ) (car1_speed : ℝ) (meeting_time : ℝ) :
  highway_length = 45 →
  car1_speed = 14 →
  meeting_time = 1.5 →
  ∃ car2_speed : ℝ,
    car2_speed = 16 ∧
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length :=
by sorry

end two_cars_meeting_l2469_246953


namespace arithmetic_sequence_and_sum_l2469_246989

def a (n : ℕ) : ℚ := 9/2 - n

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a (n + 1) - a n = -1) ∧
  (Finset.sum (Finset.range 20) a = -120) := by
  sorry

end arithmetic_sequence_and_sum_l2469_246989


namespace base12_2413_mod_9_l2469_246946

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2413
def base12_2413 : List Nat := [2, 4, 1, 3]

-- Theorem statement
theorem base12_2413_mod_9 :
  (base12ToDecimal base12_2413) % 9 = 8 := by
  sorry


end base12_2413_mod_9_l2469_246946


namespace annie_hamburger_cost_l2469_246976

/-- Calculates the cost of a single hamburger given the initial amount,
    cost of a milkshake, number of hamburgers and milkshakes bought,
    and the remaining amount after purchase. -/
def hamburger_cost (initial_amount : ℕ) (milkshake_cost : ℕ) 
                   (hamburgers_bought : ℕ) (milkshakes_bought : ℕ) 
                   (remaining_amount : ℕ) : ℕ :=
  (initial_amount - remaining_amount - milkshake_cost * milkshakes_bought) / hamburgers_bought

/-- Theorem stating that given Annie's purchases and finances, 
    each hamburger costs $4. -/
theorem annie_hamburger_cost : 
  hamburger_cost 132 5 8 6 70 = 4 := by
  sorry

end annie_hamburger_cost_l2469_246976


namespace complex_simplification_l2469_246983

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex fraction in the problem -/
noncomputable def z : ℂ := (2 + 3*i) / (2 - 3*i)

/-- The main theorem -/
theorem complex_simplification : z^8 * 3 = 3 := by sorry

end complex_simplification_l2469_246983


namespace second_vessel_ratio_l2469_246984

/-- Represents the ratio of milk to water in a mixture -/
structure MilkWaterRatio where
  milk : ℚ
  water : ℚ

/-- The mixture in a vessel -/
structure Mixture where
  volume : ℚ
  ratio : MilkWaterRatio

theorem second_vessel_ratio 
  (v1 v2 : Mixture) 
  (h1 : v1.volume = v2.volume) 
  (h2 : v1.ratio = MilkWaterRatio.mk 4 2) 
  (h3 : let combined_ratio := MilkWaterRatio.mk 
          (v1.ratio.milk * v1.volume + v2.ratio.milk * v2.volume) 
          (v1.ratio.water * v1.volume + v2.ratio.water * v2.volume)
        combined_ratio = MilkWaterRatio.mk 3 1) :
  v2.ratio = MilkWaterRatio.mk 5 7 := by
  sorry

end second_vessel_ratio_l2469_246984


namespace transformation_result_l2469_246916

-- Define the initial point
def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, z, -y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  rotate_z_90 (reflect_yz (rotate_x_90 (reflect_xz (rotate_z_90 p))))

-- Theorem statement
theorem transformation_result :
  transform initial_point = (2, -2, -2) := by sorry

end transformation_result_l2469_246916


namespace angle_r_measure_l2469_246915

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- The measure of angle P in degrees -/
  angle_p : ℝ
  /-- The measure of angle R is 40 degrees more than angle P -/
  angle_r : ℝ := angle_p + 40
  /-- The sum of all angles in the triangle is 180 degrees -/
  angle_sum : angle_p + angle_p + angle_r = 180

/-- The measure of angle R in an isosceles triangle with the given conditions -/
theorem angle_r_measure (t : IsoscelesTriangle) : t.angle_r = 86.67 := by
  sorry

end angle_r_measure_l2469_246915


namespace inequality_proof_l2469_246990

theorem inequality_proof (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : a < b) 
  (hbc : b < c) : 
  a + b < b + c := by
sorry

end inequality_proof_l2469_246990


namespace system_solution_l2469_246943

theorem system_solution (a b : ℝ) : 
  (2 * a * 1 + b * 1 = 3) → 
  (a * 1 - b * 1 = 1) → 
  a + 2 * b = 2 := by
sorry

end system_solution_l2469_246943


namespace prime_divisors_of_p_cubed_plus_three_l2469_246999

theorem prime_divisors_of_p_cubed_plus_three (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p^2 + 2)) :
  ∃ (a b c : ℕ), Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ p^3 + 3 = a * b * c :=
sorry

end prime_divisors_of_p_cubed_plus_three_l2469_246999


namespace pet_store_bird_count_l2469_246936

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_bird_count : total_birds = 72 := by
  sorry

end pet_store_bird_count_l2469_246936


namespace largest_A_value_l2469_246913

theorem largest_A_value : ∃ (A : ℝ),
  (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ A * (x - y)^2) ∧
  (∀ (B : ℝ), (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ B * (x - y)^2) → B ≤ A) ∧
  A = 18 :=
by sorry

end largest_A_value_l2469_246913


namespace hyperbola_properties_l2469_246959

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define point P on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b 2 3

-- Define the condition for slope of MA being 1 and MF = AF
def slope_and_distance_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola a b x y ∧ (y - (-a)) / (x - (-a)) = 1 ∧
  (x - 2*a)^2 + y^2 = (3*a)^2

-- Define the perpendicularity condition
def perpendicular_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  point_on_hyperbola a b →
  slope_and_distance_condition a b →
  perpendicular_condition a b →
  (a = 1 ∧ b = Real.sqrt 3) ∧
  (∀ (k t : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola 1 (Real.sqrt 3) x₁ y₁ ∧
    hyperbola 1 (Real.sqrt 3) x₂ y₂ ∧
    y₁ = k * x₁ + t ∧
    y₂ = k * x₂ + t) →
  |t| / Real.sqrt (1 + k^2) = Real.sqrt 6 / 2) :=
sorry

end hyperbola_properties_l2469_246959


namespace lottery_jackpot_probability_l2469_246980

def num_megaballs : ℕ := 30
def num_winnerballs : ℕ := 49
def num_chosen_winnerballs : ℕ := 6
def lower_sum_bound : ℕ := 100
def upper_sum_bound : ℕ := 150

def N : ℕ := sorry -- Number of ways to choose 6 numbers from 49 that sum to [100, 150]

theorem lottery_jackpot_probability :
  ∃ (p : ℚ), p = (1 : ℚ) / num_megaballs * (N : ℚ) / (Nat.choose num_winnerballs num_chosen_winnerballs) :=
by sorry

end lottery_jackpot_probability_l2469_246980


namespace georges_socks_l2469_246924

theorem georges_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ) :
  initial_socks = 28 →
  thrown_away = 4 →
  final_socks = 60 →
  final_socks - (initial_socks - thrown_away) = 36 :=
by
  sorry

end georges_socks_l2469_246924


namespace ariella_interest_rate_l2469_246923

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem ariella_interest_rate :
  ∀ (daniella_initial ariella_initial ariella_final : ℝ),
  daniella_initial = 400 →
  ariella_initial = daniella_initial + 200 →
  ariella_final = 720 →
  ∃ (rate : ℝ), 
    simple_interest ariella_initial rate 2 = ariella_final ∧
    rate = 0.1 := by
  sorry

end ariella_interest_rate_l2469_246923


namespace transaction_difference_l2469_246932

theorem transaction_difference : 
  ∀ (mabel anthony cal jade : ℕ),
    mabel = 90 →
    anthony = mabel + mabel / 10 →
    cal = (2 * anthony) / 3 →
    jade = 81 →
    jade - cal = 15 :=
by
  sorry

end transaction_difference_l2469_246932


namespace complement_A_in_U_l2469_246972

open Set

def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def U : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

end complement_A_in_U_l2469_246972


namespace find_y_value_l2469_246950

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end find_y_value_l2469_246950


namespace perfect_square_trinomial_k_l2469_246928

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: If x^2 - kx + 64 is a perfect square trinomial, then k = 16 or k = -16 -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 64 → k = 16 ∨ k = -16 := by
  sorry


end perfect_square_trinomial_k_l2469_246928


namespace energy_after_moving_charge_l2469_246939

/-- The energy stored between two point charges is inversely proportional to their distance -/
axiom energy_inverse_distance {d₁ d₂ : ℝ} {E₁ E₂ : ℝ} (h : d₁ > 0 ∧ d₂ > 0) :
  E₁ / E₂ = d₂ / d₁

/-- The total energy of four point charges at the corners of a square -/
def initial_energy : ℝ := 20

/-- The number of energy pairs in the initial square configuration -/
def initial_pairs : ℕ := 6

theorem energy_after_moving_charge (d : ℝ) (h : d > 0) :
  let initial_pair_energy := initial_energy / initial_pairs
  let center_to_corner_distance := d / Real.sqrt 2
  let new_center_pair_energy := initial_pair_energy * d / center_to_corner_distance
  3 * new_center_pair_energy + 3 * initial_pair_energy = 10 * Real.sqrt 2 + 10 := by
sorry

end energy_after_moving_charge_l2469_246939


namespace problem_solution_l2469_246938

theorem problem_solution (t : ℝ) :
  let x := 3 - 2*t
  let y := t^2 + 3*t + 6
  x = -1 → y = 16 := by
sorry

end problem_solution_l2469_246938


namespace alphabet_value_problem_l2469_246998

theorem alphabet_value_problem (H M A T E : ℤ) : 
  H = 8 →
  M + A + T + H = 31 →
  T + E + A + M = 40 →
  M + E + E + T = 44 →
  M + A + T + E = 39 →
  A = 12 := by
sorry

end alphabet_value_problem_l2469_246998


namespace remainder_theorem_l2469_246996

theorem remainder_theorem : ∃ q : ℕ, 3^202 + 303 = (3^101 + 3^51 + 1) * q + 302 := by
  sorry

end remainder_theorem_l2469_246996


namespace mans_speed_against_current_l2469_246991

/-- 
Given a man's speed with the current and the speed of the current,
this theorem proves the man's speed against the current.
-/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 3) : 
  speed_with_current - 2 * current_speed = 14 :=
by sorry

end mans_speed_against_current_l2469_246991


namespace square_sum_divisors_l2469_246968

theorem square_sum_divisors (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 5 ∨ n = 8 ∨ n = 20 := by
sorry

end square_sum_divisors_l2469_246968


namespace blackboard_problem_l2469_246914

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing a set of numbers with their sum modulo m -/
def replace_with_sum_mod (m : ℕ) (s : Finset ℕ) : ℕ := (s.sum id) % m

theorem blackboard_problem :
  ∀ (s : Finset ℕ),
  s.card = 2 →
  999 ∈ s →
  (∃ (t : Finset ℕ), t.card = 2004 ∧ Finset.range 2004 = t ∧
   replace_with_sum_mod 167 t = replace_with_sum_mod 167 s) →
  ∃ x, x ∈ s ∧ x ≠ 999 ∧ x = 3 := by
  sorry

#check blackboard_problem

end blackboard_problem_l2469_246914


namespace second_class_size_l2469_246970

/-- Given two classes of students, prove that the second class has 50 students. -/
theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℝ) 
  (second_class_avg : ℝ) (total_avg : ℝ) :
  first_class_size = 30 →
  first_class_avg = 30 →
  second_class_avg = 60 →
  total_avg = 48.75 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size : ℝ) = total_avg ∧
    second_class_size = 50 :=
by sorry

end second_class_size_l2469_246970


namespace corner_rectangles_area_sum_l2469_246973

/-- Given a square with side length 100 cm divided into 9 rectangles,
    where the central rectangle has dimensions 40 cm × 60 cm,
    the sum of the areas of the four corner rectangles is 2400 cm². -/
theorem corner_rectangles_area_sum (x y : ℝ) : x > 0 → y > 0 →
  x + 40 + (100 - x - 40) = 100 →
  y + 60 + (100 - y - 60) = 100 →
  x * y + (60 - x) * y + x * (40 - y) + (60 - x) * (40 - y) = 2400 := by
  sorry

#check corner_rectangles_area_sum

end corner_rectangles_area_sum_l2469_246973


namespace greatest_prime_factor_of_147_l2469_246945

theorem greatest_prime_factor_of_147 : ∃ p : ℕ, p = 7 ∧ Nat.Prime p ∧ p ∣ 147 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 147 → q ≤ p :=
  sorry

end greatest_prime_factor_of_147_l2469_246945


namespace distribute_5_4_l2469_246954

/-- The number of ways to distribute n different books to k students,
    with each student receiving at least one book. -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n - (k.choose 3) * (k-3)^n

/-- Theorem stating that distributing 5 different books to 4 students,
    with each student receiving at least one book, results in 240 different schemes. -/
theorem distribute_5_4 : distribute 5 4 = 240 := by
  sorry

end distribute_5_4_l2469_246954


namespace specific_cube_surface_area_l2469_246947

/-- Represents a cube with circular holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_diameter : ℝ

/-- Calculates the total surface area of a cube with circular holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  sorry

/-- Theorem stating the total surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_diameter := 2 }
  total_surface_area cube = 96 + 42 * Real.pi := by
  sorry

end specific_cube_surface_area_l2469_246947


namespace square_divided_into_rectangles_l2469_246956

theorem square_divided_into_rectangles (square_perimeter : ℝ) 
  (h1 : square_perimeter = 200) : 
  let side_length := square_perimeter / 4
  let rectangle_length := side_length
  let rectangle_width := side_length / 2
  2 * (rectangle_length + rectangle_width) = 150 := by
sorry

end square_divided_into_rectangles_l2469_246956


namespace ellipse_major_axis_length_l2469_246997

/-- Given an ellipse C with equation x^2/4 + y^2/m^2 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2/4 + y^2/m^2 = 1}
  let focal_length : ℝ := 4
  ∃ (major_axis_length : ℝ), major_axis_length = 4 * Real.sqrt 2 := by
  sorry

end ellipse_major_axis_length_l2469_246997


namespace billys_age_l2469_246965

theorem billys_age (billy brenda joe : ℚ) 
  (h1 : billy = 3 * brenda)
  (h2 : billy = 2 * joe)
  (h3 : billy + brenda + joe = 72) :
  billy = 432 / 11 := by
sorry

end billys_age_l2469_246965


namespace valid_table_iff_odd_l2469_246957

/-- A square table of size n × n -/
def SquareTable (n : ℕ) := Fin n → Fin n → ℚ

/-- The sum of numbers on a diagonal of a square table -/
def diagonalSum (table : SquareTable n) (d : ℕ) : ℚ :=
  sorry

/-- A square table is valid if the sum of numbers on each diagonal is 1 -/
def isValidTable (table : SquareTable n) : Prop :=
  ∀ d, d < 4*n - 2 → diagonalSum table d = 1

/-- There exists a valid square table of size n × n if and only if n is odd -/
theorem valid_table_iff_odd (n : ℕ) :
  (∃ (table : SquareTable n), isValidTable table) ↔ Odd n :=
sorry

end valid_table_iff_odd_l2469_246957


namespace rectangle_area_preservation_l2469_246910

theorem rectangle_area_preservation (L W : ℝ) (x : ℝ) (h : x > 0) :
  L * W = L * (1 - x / 100) * W * (1 + 11.111111111111107 / 100) →
  x = 10 := by
sorry

end rectangle_area_preservation_l2469_246910


namespace book_club_members_count_l2469_246907

def annual_snack_fee : ℕ := 150
def hardcover_books_count : ℕ := 6
def hardcover_book_price : ℕ := 30
def paperback_books_count : ℕ := 6
def paperback_book_price : ℕ := 12
def total_collected : ℕ := 2412

theorem book_club_members_count :
  let cost_per_member := annual_snack_fee +
    hardcover_books_count * hardcover_book_price +
    paperback_books_count * paperback_book_price
  total_collected / cost_per_member = 6 :=
by sorry

end book_club_members_count_l2469_246907


namespace line_slope_intercept_product_l2469_246985

theorem line_slope_intercept_product (m b : ℚ) (h1 : m = 1/3) (h2 : b = -3/4) :
  -1 < m * b ∧ m * b < 0 := by sorry

end line_slope_intercept_product_l2469_246985


namespace volleyball_team_selection_l2469_246987

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 31460 := by
  sorry

end volleyball_team_selection_l2469_246987


namespace tony_investment_rate_l2469_246917

/-- Calculates the investment rate given the investment amount and annual income. -/
def investment_rate (investment : ℚ) (annual_income : ℚ) : ℚ :=
  (annual_income / investment) * 100

/-- Proves that the investment rate is 7.8125% for the given scenario. -/
theorem tony_investment_rate :
  let investment := 3200
  let annual_income := 250
  investment_rate investment annual_income = 7.8125 := by
  sorry

end tony_investment_rate_l2469_246917


namespace cube_tetrahedron_surface_area_ratio_l2469_246903

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length^2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length^2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
sorry

end cube_tetrahedron_surface_area_ratio_l2469_246903


namespace range_of_m_for_subset_l2469_246930

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m}

-- The main theorem
theorem range_of_m_for_subset (h : ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ∧ 
  ¬(∀ x : ℝ, x ∈ B m → x ∈ A)) : 
  {m : ℝ | A ⊆ B m} = {m : ℝ | m > 3} := by
  sorry


end range_of_m_for_subset_l2469_246930


namespace derivative_of_f_l2469_246901

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

theorem derivative_of_f (x : ℝ) (h : x > 0) : 
  deriv f x = (x^(Real.sqrt 2) / (2 * Real.sqrt 2)) * 
    (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x)) := by
  sorry

end derivative_of_f_l2469_246901


namespace hot_dog_price_l2469_246931

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = 10

/-- Second day's purchase equation -/
axiom day2_equation : 2 * hamburger_cost + 3 * hot_dog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hot_dog_price : hot_dog_cost = 1 := by sorry

end hot_dog_price_l2469_246931


namespace min_value_c_l2469_246908

-- Define the consecutive integers
def consecutive_integers (a b c d e : ℕ) : Prop :=
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e

-- Define perfect square and perfect cube
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

-- Main theorem
theorem min_value_c (a b c d e : ℕ) :
  consecutive_integers a b c d e →
  is_perfect_square (b + c + d) →
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ (∀ c' : ℕ, c' < 675 → 
    ¬(∃ a' b' d' e' : ℕ, consecutive_integers a' b' c' d' e' ∧
      is_perfect_square (b' + c' + d') ∧
      is_perfect_cube (a' + b' + c' + d' + e'))) :=
by sorry

end min_value_c_l2469_246908


namespace regular_soda_bottles_l2469_246934

theorem regular_soda_bottles (diet_soda : ℕ) (difference : ℕ) : 
  diet_soda = 19 → difference = 41 → diet_soda + difference = 60 :=
by
  sorry

end regular_soda_bottles_l2469_246934


namespace polynomial_factor_implies_a_minus_b_eq_one_l2469_246942

/-- The polynomial in question -/
def P (a b x y : ℝ) : ℝ := x^2 + a*x*y + b*y^2 - 5*x + y + 6

/-- The factor of the polynomial -/
def F (x y : ℝ) : ℝ := x + y - 2

theorem polynomial_factor_implies_a_minus_b_eq_one (a b : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, P a b x y = F x y * k) →
  a - b = 1 := by sorry

end polynomial_factor_implies_a_minus_b_eq_one_l2469_246942


namespace loan_duration_proof_l2469_246905

/-- Represents the annual interest rate as a decimal -/
def interest_rate (percent : ℚ) : ℚ := percent / 100

/-- Calculates the annual interest given a principal and an interest rate -/
def annual_interest (principal : ℚ) (rate : ℚ) : ℚ := principal * rate

/-- Calculates the gain over a period of time -/
def gain (annual_gain : ℚ) (years : ℚ) : ℚ := annual_gain * years

theorem loan_duration_proof (principal : ℚ) (rate_A_to_B rate_B_to_C total_gain : ℚ) :
  principal = 2000 →
  rate_A_to_B = interest_rate 10 →
  rate_B_to_C = interest_rate 11.5 →
  total_gain = 90 →
  gain (annual_interest principal rate_B_to_C - annual_interest principal rate_A_to_B) 3 = total_gain :=
by sorry

end loan_duration_proof_l2469_246905


namespace no_injective_function_exists_l2469_246962

theorem no_injective_function_exists : ¬∃ f : ℝ → ℝ, 
  Function.Injective f ∧ ∀ x : ℝ, f (x^2) - (f x)^2 ≥ (1/4 : ℝ) := by
  sorry

end no_injective_function_exists_l2469_246962


namespace prob_three_even_d20_l2469_246971

/-- A fair 20-sided die -/
def D20 : Type := Fin 20

/-- The probability of rolling an even number on a D20 -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The number of dice showing even numbers -/
def k : ℕ := 3

/-- The probability of rolling exactly k even numbers out of n rolls -/
def prob_k_even (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem prob_three_even_d20 :
  prob_k_even n k prob_even = 5/16 := by
  sorry

end prob_three_even_d20_l2469_246971


namespace five_in_set_A_l2469_246955

theorem five_in_set_A : 5 ∈ {x : ℕ | 1 ≤ x ∧ x ≤ 5} := by
  sorry

end five_in_set_A_l2469_246955


namespace clothing_store_problem_l2469_246974

-- Define the types of clothing
inductive ClothingType
| A
| B

-- Define the structure for clothing information
structure ClothingInfo where
  purchasePrice : ClothingType → ℕ
  sellingPrice : ClothingType → ℕ
  totalQuantity : ℕ

-- Define the problem conditions
def problemConditions (info : ClothingInfo) : Prop :=
  info.totalQuantity = 100 ∧
  2 * info.purchasePrice ClothingType.A + info.purchasePrice ClothingType.B = 260 ∧
  info.purchasePrice ClothingType.A + 3 * info.purchasePrice ClothingType.B = 380 ∧
  info.sellingPrice ClothingType.A = 120 ∧
  info.sellingPrice ClothingType.B = 150

-- Define the profit calculation function
def calculateProfit (info : ClothingInfo) (quantityA quantityB : ℕ) : ℕ :=
  (info.sellingPrice ClothingType.A - info.purchasePrice ClothingType.A) * quantityA +
  (info.sellingPrice ClothingType.B - info.purchasePrice ClothingType.B) * quantityB

-- Theorem statement
theorem clothing_store_problem (info : ClothingInfo) :
  problemConditions info →
  (info.purchasePrice ClothingType.A = 80 ∧
   info.purchasePrice ClothingType.B = 100 ∧
   calculateProfit info 50 50 = 4500 ∧
   (∀ m : ℕ, m ≤ 33 → (100 - m) ≥ 2 * m) ∧
   (∀ m : ℕ, m > 33 → (100 - m) < 2 * m) ∧
   calculateProfit info 67 33 = 4330) :=
by sorry


end clothing_store_problem_l2469_246974


namespace sally_grew_six_carrots_l2469_246961

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end sally_grew_six_carrots_l2469_246961


namespace average_pastry_sales_l2469_246911

/-- Represents the daily sales of pastries over a week -/
def weeklySales : List Nat := [2, 3, 4, 5, 6, 7, 8]

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Calculates the average of a list of natural numbers -/
def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem average_pastry_sales : average weeklySales = 5 := by sorry

end average_pastry_sales_l2469_246911


namespace inequality_proof_l2469_246922

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end inequality_proof_l2469_246922


namespace min_value_theorem_l2469_246920

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2) + y^3 / (x - 2)) ≥ 54 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ x₀^3 / (y₀ - 2) + y₀^3 / (x₀ - 2) = 54 :=
by sorry

end min_value_theorem_l2469_246920


namespace unique_three_digit_number_l2469_246902

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def switch_outermost_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem unique_three_digit_number : 
  ∃! n : ℕ, is_three_digit n ∧ n + 1 = 2 * (switch_outermost_digits n) :=
by
  use 793
  sorry

#eval switch_outermost_digits 793
#eval 793 + 1 = 2 * (switch_outermost_digits 793)

end unique_three_digit_number_l2469_246902


namespace quadratic_coefficient_l2469_246925

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) := x^2 - c*x + 6

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x > 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_coefficient : ∃ c : ℝ, condition c ∧ c = 1 := by
  sorry

end quadratic_coefficient_l2469_246925


namespace factor_calculation_l2469_246940

theorem factor_calculation (original_number : ℝ) (final_result : ℝ) : 
  original_number = 7 →
  final_result = 69 →
  ∃ (factor : ℝ), factor * (2 * original_number + 9) = final_result ∧ factor = 3 :=
by sorry

end factor_calculation_l2469_246940


namespace combined_tax_rate_l2469_246929

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end combined_tax_rate_l2469_246929


namespace tau_prime_factors_divide_l2469_246918

/-- The number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- For positive integers a and b, if σ(a^n) divides σ(b^n) for all n ∈ ℕ,
    then each prime factor of τ(a) divides τ(b) -/
theorem tau_prime_factors_divide (a b : ℕ+) 
  (h : ∀ n : ℕ, (sigma (a^n) : ℕ) ∣ (sigma (b^n) : ℕ)) :
  ∀ p : ℕ, Prime p → p ∣ tau a → p ∣ tau b := by
  sorry

end tau_prime_factors_divide_l2469_246918


namespace button_comparison_l2469_246969

theorem button_comparison (mari_buttons sue_buttons : ℕ) 
  (h1 : mari_buttons = 8)
  (h2 : sue_buttons = 22)
  (h3 : ∃ kendra_buttons : ℕ, sue_buttons = kendra_buttons / 2)
  (h4 : ∃ kendra_buttons : ℕ, kendra_buttons > 5 * mari_buttons) :
  ∃ kendra_buttons : ℕ, kendra_buttons - (5 * mari_buttons) = 4 :=
by sorry

end button_comparison_l2469_246969


namespace pudding_distribution_l2469_246951

theorem pudding_distribution (total_cups : ℕ) (additional_cups : ℕ) 
  (h1 : total_cups = 315)
  (h2 : additional_cups = 121)
  (h3 : ∀ (students : ℕ), students > 0 → 
    (total_cups + additional_cups) % students = 0 → 
    total_cups < students * ((total_cups + additional_cups) / students)) :
  ∃ (students : ℕ), students = 4 ∧ 
    (total_cups + additional_cups) % students = 0 ∧
    total_cups < students * ((total_cups + additional_cups) / students) :=
by sorry

end pudding_distribution_l2469_246951


namespace franks_sunday_bags_l2469_246944

/-- Given that Frank filled 5 bags on Saturday, each bag contains 5 cans,
    and Frank collected a total of 40 cans over the weekend,
    prove that Frank filled 3 bags on Sunday. -/
theorem franks_sunday_bags (saturday_bags : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ)
    (h1 : saturday_bags = 5)
    (h2 : cans_per_bag = 5)
    (h3 : total_cans = 40) :
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 := by
  sorry

end franks_sunday_bags_l2469_246944


namespace small_bottle_price_approx_l2469_246986

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((average_price * (large_quantity + small_quantity : ℚ)) - (large_quantity : ℚ) * large_price) / (small_quantity : ℚ)

/-- Theorem stating that the price of small bottles is approximately $1.38 -/
theorem small_bottle_price_approx :
  let large_quantity : ℕ := 1300
  let small_quantity : ℕ := 750
  let large_price : ℚ := 189/100
  let average_price : ℚ := 17034/10000
  let calculated_price := price_small_bottle large_quantity small_quantity large_price average_price
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |calculated_price - 138/100| < ε :=
sorry

end small_bottle_price_approx_l2469_246986
