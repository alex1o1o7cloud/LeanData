import Mathlib

namespace sequence_sum_l2177_217734

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum (a b : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 3 * a 11 = 4 * a 7) →                              -- given condition for geometric sequence
  (∀ n : ℕ, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- arithmetic sequence condition
  (a 7 = b 7) →                                         -- given condition relating both sequences
  b 5 + b 9 = 8 := by
sorry

end sequence_sum_l2177_217734


namespace vector_parallel_condition_l2177_217755

/-- Given vectors in ℝ², prove that if a + 3b is parallel to c, then m = -6 -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) (m : ℝ) 
  (ha : a = (-2, 3))
  (hb : b = (3, 1))
  (hc : c = (-7, m))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • c) :
  m = -6 := by
  sorry

end vector_parallel_condition_l2177_217755


namespace solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2177_217703

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x + 1| - |x - 1|

-- Part 1
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 3/2} = {x : ℝ | x < 3/4} := by sorry

-- Part 2
theorem range_of_m_when_a_gt_one (a : ℝ) (m : ℝ) :
  a > 1 →
  (∃ x : ℝ, f a x ≤ -|2*m + 1|) →
  m ∈ Set.Icc (-3/2) 1 := by sorry

end solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2177_217703


namespace amount_to_hand_in_l2177_217722

/-- Represents the denominations of bills in US currency --/
inductive Denomination
  | Hundred
  | Fifty
  | Twenty
  | Ten
  | Five
  | One

/-- Represents the quantity of each denomination in the till --/
def till_contents : Denomination → ℕ
  | Denomination.Hundred => 2
  | Denomination.Fifty => 1
  | Denomination.Twenty => 5
  | Denomination.Ten => 3
  | Denomination.Five => 7
  | Denomination.One => 27

/-- The value of each denomination in dollars --/
def denomination_value : Denomination → ℕ
  | Denomination.Hundred => 100
  | Denomination.Fifty => 50
  | Denomination.Twenty => 20
  | Denomination.Ten => 10
  | Denomination.Five => 5
  | Denomination.One => 1

/-- The amount to be left in the till --/
def amount_to_leave : ℕ := 300

/-- Calculates the total value of bills in the till --/
def total_value : ℕ := sorry

/-- Theorem: The amount Jack will hand in is $142 --/
theorem amount_to_hand_in :
  total_value - amount_to_leave = 142 := by sorry

end amount_to_hand_in_l2177_217722


namespace meaningful_expression_l2177_217744

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end meaningful_expression_l2177_217744


namespace third_jumper_height_l2177_217766

/-- The height of Ravi's jump in inches -/
def ravi_jump : ℝ := 39

/-- The height of the first next highest jumper in inches -/
def jumper1 : ℝ := 23

/-- The height of the second next highest jumper in inches -/
def jumper2 : ℝ := 27

/-- The factor by which Ravi can jump higher than the average of the next three highest jumpers -/
def ravi_factor : ℝ := 1.5

/-- The height of the third next highest jumper in inches -/
def jumper3 : ℝ := 28

theorem third_jumper_height :
  ravi_jump = ravi_factor * ((jumper1 + jumper2 + jumper3) / 3) :=
by sorry

end third_jumper_height_l2177_217766


namespace paper_airplane_competition_l2177_217716

theorem paper_airplane_competition
  (a b h v m : ℝ)
  (total : a + b + h + v + m = 41)
  (matyas_least : m ≤ a ∧ m ≤ b ∧ m ≤ h ∧ m ≤ v)
  (andelka_matyas : a = m + 0.9)
  (vlada_andelka : v = a + 0.6)
  (honzik_furthest : h > a ∧ h > b ∧ h > v ∧ h > m)
  (honzik_whole : ∃ n : ℕ, h = n)
  (avg_difference : (a + v + m) / 3 = (a + b + h + v + m) / 5 - 0.2) :
  a = 8.1 ∧ b = 8 ∧ h = 9 ∧ v = 8.7 ∧ m = 7.2 := by
sorry

end paper_airplane_competition_l2177_217716


namespace transformed_square_properties_l2177_217774

/-- A point in the xy-plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The transformation from xy-plane to uv-plane -/
def transform (p : Point) : Point :=
  { x := p.x^2 + p.y^2,
    y := p.x^2 * p.y^2 }

/-- The unit square PQRST in the xy-plane -/
def unitSquare : Set Point :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- The image of the unit square under the transformation -/
def transformedSquare : Set Point :=
  {q | ∃ p ∈ unitSquare, q = transform p}

/-- Definition of vertical symmetry -/
def verticallySymmetric (s : Set Point) : Prop :=
  ∀ p ∈ s, ∃ q ∈ s, q.x = p.x ∧ q.y = -p.y

/-- Definition of curved upper boundary -/
def hasCurvedUpperBoundary (s : Set Point) : Prop :=
  ∃ f : ℝ → ℝ, (∀ x, f x ≥ 0) ∧ 
    (∀ p ∈ s, p.y ≤ f p.x) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂)

theorem transformed_square_properties :
  verticallySymmetric transformedSquare ∧ 
  hasCurvedUpperBoundary transformedSquare :=
sorry

end transformed_square_properties_l2177_217774


namespace coefficient_x_squared_in_x_minus_one_to_eighth_l2177_217751

theorem coefficient_x_squared_in_x_minus_one_to_eighth (x : ℝ) : 
  (∃ a b c d e f g : ℝ, (x - 1)^8 = x^8 + 8*x^7 + 28*x^6 + 56*x^5 + 70*x^4 + a*x^3 + b*x^2 + c*x + d) ∧ 
  (∃ p q r s t u v : ℝ, (x - 1)^8 = p*x^7 + q*x^6 + r*x^5 + s*x^4 + t*x^3 + 28*x^2 + u*x + v) :=
by sorry

end coefficient_x_squared_in_x_minus_one_to_eighth_l2177_217751


namespace operation_with_96_percent_error_l2177_217776

/-- Given a number N and an operation O(N), if the percentage error between O(N) and 5N is 96%, then O(N) = 0.2N -/
theorem operation_with_96_percent_error (N : ℝ) (O : ℝ → ℝ) :
  (|O N - 5 * N| / (5 * N) = 0.96) → O N = 0.2 * N :=
by sorry

end operation_with_96_percent_error_l2177_217776


namespace quadratic_sum_l2177_217730

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c) →
  a + b + c = 353/4 := by
sorry

end quadratic_sum_l2177_217730


namespace log_problem_l2177_217731

theorem log_problem (m : ℝ) : 
  (Real.log 4 / Real.log 3) * (Real.log 8 / Real.log 4) * (Real.log m / Real.log 8) = Real.log 16 / Real.log 4 → 
  m = 9 := by
sorry

end log_problem_l2177_217731


namespace speed_ratio_is_two_l2177_217728

/-- Given a round trip with total distance 60 km, total time 6 hours, and return speed 15 km/h,
    the ratio of return speed to outbound speed is 2. -/
theorem speed_ratio_is_two 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_distance = 60) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 15) : 
  return_speed / ((total_distance / 2) / (total_time - total_distance / (2 * return_speed))) = 2 := by
  sorry

end speed_ratio_is_two_l2177_217728


namespace stuffed_animals_theorem_l2177_217793

/-- Given the number of stuffed animals for McKenna (M), Kenley (K), and Tenly (T),
    prove various properties about their stuffed animal collection. -/
theorem stuffed_animals_theorem (M K T : ℕ) (S : ℕ) (A F : ℚ) 
    (hM : M = 34)
    (hK : K = 2 * M)
    (hT : T = K + 5)
    (hS : S = M + K + T)
    (hA : A = S / 3)
    (hF : F = M / S) : 
  K = 68 ∧ 
  T = 73 ∧ 
  S = 175 ∧ 
  A = 175 / 3 ∧ 
  F = 34 / 175 := by
  sorry

end stuffed_animals_theorem_l2177_217793


namespace line_intercept_sum_l2177_217780

/-- Given a line 5x + 8y + c = 0, if the sum of its x-intercept and y-intercept is 26, then c = -80 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ x y : ℝ, 5*x + 8*y + c = 0 ∧ 5*x + c = 0 ∧ 8*y + c = 0 ∧ x + y = 26) → 
  c = -80 :=
by sorry

end line_intercept_sum_l2177_217780


namespace set_intersection_equality_l2177_217739

-- Define set A
def A : Set ℝ := {y | ∃ x > 1, y = Real.log x}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem set_intersection_equality : (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end set_intersection_equality_l2177_217739


namespace train_length_l2177_217702

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  crossing_time = 30 → 
  bridge_length = 350 → 
  train_speed * crossing_time - bridge_length = 250 := by sorry

end train_length_l2177_217702


namespace quadratic_equations_solutions_l2177_217783

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + 2 * Real.sqrt 2 ∧ x₂ = 3 - 2 * Real.sqrt 2 ∧
    x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1 ∧ y₂ = 1/2 ∧
    2*(y₁+1)^2 = 3*(y₁+1) ∧ 2*(y₂+1)^2 = 3*(y₂+1)) :=
by sorry

end quadratic_equations_solutions_l2177_217783


namespace relationship_xyz_l2177_217784

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = Real.rpow 0.5 0.5)
  (hy : y = Real.rpow 0.5 1.3)
  (hz : z = Real.rpow 1.3 0.5) :
  y < x ∧ x < z := by
  sorry

end relationship_xyz_l2177_217784


namespace find_Y_value_l2177_217705

theorem find_Y_value (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 := by
  sorry

end find_Y_value_l2177_217705


namespace dog_bath_time_l2177_217723

/-- Represents the time spent on various activities with a dog -/
structure DogCareTime where
  total : ℝ
  walking : ℝ
  bath : ℝ
  blowDry : ℝ

/-- Represents the walking parameters -/
structure WalkingParams where
  distance : ℝ
  speed : ℝ

/-- Theorem stating the bath time given the conditions -/
theorem dog_bath_time (t : DogCareTime) (w : WalkingParams) : 
  t.total = 60 ∧ 
  w.distance = 3 ∧ 
  w.speed = 6 ∧ 
  t.blowDry = t.bath / 2 ∧ 
  t.total = t.walking + t.bath + t.blowDry ∧ 
  t.walking = w.distance / w.speed * 60 →
  t.bath = 20 := by
  sorry


end dog_bath_time_l2177_217723


namespace equilateral_triangle_revolution_surface_area_l2177_217775

/-- The surface area of a solid of revolution formed by rotating an equilateral triangle -/
theorem equilateral_triangle_revolution_surface_area 
  (side_length : ℝ) 
  (h_side : side_length = 2) : 
  let solid_surface_area := 2 * Real.pi * (side_length * Real.sqrt 3 / 2) * side_length
  solid_surface_area = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end equilateral_triangle_revolution_surface_area_l2177_217775


namespace cosA_value_l2177_217712

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosA_value (t : Triangle) 
  (h : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C) : 
  Real.cos t.A = 1/2 := by
  sorry

end cosA_value_l2177_217712


namespace missing_digit_is_five_l2177_217721

def largest_number (x : ℕ) : ℕ :=
  if x ≥ 2 then 9000 + 100 * x + 21 else 9000 + 200 + x

def smallest_number (x : ℕ) : ℕ := 1000 + 200 + 90 + x

theorem missing_digit_is_five :
  ∀ x : ℕ, x < 10 →
    largest_number x - smallest_number x = 8262 →
    x = 5 := by
  sorry

end missing_digit_is_five_l2177_217721


namespace polynomial_identity_sum_of_squares_l2177_217773

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 1000 * x^3 + 27 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end polynomial_identity_sum_of_squares_l2177_217773


namespace simplify_sqrt_expression_l2177_217779

theorem simplify_sqrt_expression : 
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l2177_217779


namespace joes_juices_l2177_217782

/-- The number of juices Joe bought at the market -/
def num_juices : ℕ := 7

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := 4.5

/-- The cost of a single juice in dollars -/
def juice_cost : ℚ := 0.5

/-- The cost of a single jar of honey in dollars -/
def honey_cost : ℚ := 5

/-- The cost of two plants in dollars -/
def two_plants_cost : ℚ := 18

/-- The total amount Joe spent at the market in dollars -/
def total_spent : ℚ := 68

/-- The number of oranges Joe bought -/
def num_oranges : ℕ := 3

/-- The number of jars of honey Joe bought -/
def num_honey : ℕ := 3

/-- The number of plants Joe bought -/
def num_plants : ℕ := 4

theorem joes_juices :
  num_juices * juice_cost = 
    total_spent - (num_oranges * orange_cost + num_honey * honey_cost + (num_plants / 2) * two_plants_cost) :=
by sorry

end joes_juices_l2177_217782


namespace log_3_infinite_sum_equals_4_l2177_217724

theorem log_3_infinite_sum_equals_4 :
  ∃ (x : ℝ), x > 0 ∧ 3^x = x + 81 ∧ x = 4 := by
  sorry

end log_3_infinite_sum_equals_4_l2177_217724


namespace missing_number_equation_l2177_217794

theorem missing_number_equation (x : ℤ) : 10010 - 12 * 3 * x = 9938 ↔ x = 2 := by
  sorry

end missing_number_equation_l2177_217794


namespace charitable_distribution_result_l2177_217701

def charitable_distribution (initial : ℕ) : ℕ :=
  let to_farmer := initial / 2 + 1
  let after_farmer := initial - to_farmer
  let to_beggar := after_farmer / 2 + 2
  let after_beggar := after_farmer - to_beggar
  let to_boy := after_beggar / 2 + 3
  after_beggar - to_boy

theorem charitable_distribution_result :
  charitable_distribution 42 = 1 := by
  sorry

end charitable_distribution_result_l2177_217701


namespace mean_equality_implies_z_l2177_217785

def mean (list : List ℚ) : ℚ := (list.sum) / list.length

theorem mean_equality_implies_z (z : ℚ) : 
  mean [7, 10, 15, 21] = mean [18, z] → z = 8.5 := by
  sorry

end mean_equality_implies_z_l2177_217785


namespace inequality_proof_l2177_217791

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a^2 + b^2 + c^2 + (a+b+c)^2 ≤ 4) :
  (a*b + 1) / (a+b)^2 + (b*c + 1) / (b+c)^2 + (c*a + 1) / (c+a)^2 ≥ 3 := by
  sorry

end inequality_proof_l2177_217791


namespace fraction_value_l2177_217713

theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  x * z / (y * w) = 20 := by
sorry

end fraction_value_l2177_217713


namespace magazine_cover_theorem_l2177_217770

theorem magazine_cover_theorem (n : ℕ) (S : ℝ) (h1 : n = 15) (h2 : S > 0) :
  ∃ (remaining_area : ℝ), remaining_area ≥ (8 / 15) * S ∧
  ∃ (remaining_magazines : ℕ), remaining_magazines = n - 7 :=
by
  sorry

end magazine_cover_theorem_l2177_217770


namespace perpendicular_slope_l2177_217764

theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℚ) (hx : x₁ ≠ x₂) :
  let m₁ := (y₂ - y₁) / (x₂ - x₁)
  let m₂ := -1 / m₁
  x₁ = 3 ∧ y₁ = -3 ∧ x₂ = -4 ∧ y₂ = 2 → m₂ = 7/5 := by
  sorry

end perpendicular_slope_l2177_217764


namespace units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l2177_217796

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the units digit of a power of 5
def unitsDigitPowerOf5 (n : ℕ) : ℕ := 5

-- Define a function to get the units digit of a power of 3
def unitsDigitPowerOf3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case should never occur

theorem units_digit_of_sum (a b : ℕ) :
  unitsDigit (a + b) = unitsDigit (unitsDigit a + unitsDigit b) :=
sorry

theorem units_digit_of_35_power_87_plus_93_power_49 :
  unitsDigit ((35 ^ 87) + (93 ^ 49)) = 8 :=
sorry

end units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l2177_217796


namespace largest_decimal_l2177_217709

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.997 → b = 0.9797 → c = 0.97 → d = 0.979 → e = 0.9709 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end largest_decimal_l2177_217709


namespace appended_ages_digits_l2177_217745

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def append_numbers (a b : ℕ) : ℕ := a * 100 + b

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem appended_ages_digits (j a : ℕ) :
  is_two_digit j →
  is_two_digit a →
  is_perfect_square (append_numbers j a) →
  digit_sum (append_numbers j a) = 7 →
  ∃ n : ℕ, append_numbers j a = n ∧ 1000 ≤ n ∧ n ≤ 9999 :=
sorry

end appended_ages_digits_l2177_217745


namespace computer_speed_significant_figures_l2177_217754

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (n : ScientificNotation) : ℕ :=
  sorry

/-- The given computer speed in scientific notation -/
def computerSpeed : ScientificNotation :=
  { coefficient := 2.09
    exponent := 10 }

/-- Theorem stating that the computer speed has 3 significant figures -/
theorem computer_speed_significant_figures :
  countSignificantFigures computerSpeed = 3 := by
  sorry

end computer_speed_significant_figures_l2177_217754


namespace jellybean_distribution_l2177_217769

theorem jellybean_distribution (total_jellybeans : ℕ) (total_recipients : ℕ) 
  (h1 : total_jellybeans = 70) (h2 : total_recipients = 5) :
  total_jellybeans / total_recipients = 14 := by
  sorry

end jellybean_distribution_l2177_217769


namespace hyperbola_m_value_l2177_217706

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop := x^2 - y^2/m^2 = 1

-- Define the condition that the conjugate axis is twice the transverse axis
def conjugate_twice_transverse (m : ℝ) : Prop := abs m = 2

-- State the theorem
theorem hyperbola_m_value : 
  ∀ m : ℝ, (∃ x y : ℝ, hyperbola_equation x y m) → conjugate_twice_transverse m → m = 2 ∨ m = -2 :=
by sorry

end hyperbola_m_value_l2177_217706


namespace range_of_a_l2177_217711

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end range_of_a_l2177_217711


namespace frac_2023rd_digit_l2177_217717

-- Define the fraction
def frac : ℚ := 7 / 26

-- Define the length of the repeating decimal
def repeat_length : ℕ := 6

-- Define the position we're interested in
def position : ℕ := 2023

-- Define the function that returns the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem frac_2023rd_digit :
  nth_digit position = 5 :=
sorry

end frac_2023rd_digit_l2177_217717


namespace stamp_cost_l2177_217758

/-- The cost of stamps problem -/
theorem stamp_cost (cost_per_stamp : ℕ) (num_stamps : ℕ) : 
  cost_per_stamp = 34 → num_stamps = 4 → cost_per_stamp * num_stamps = 136 := by
  sorry

end stamp_cost_l2177_217758


namespace necessary_not_sufficient_condition_l2177_217700

theorem necessary_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ 
  (∃ y, y ∈ C ∧ y ∉ A) := by
  sorry

end necessary_not_sufficient_condition_l2177_217700


namespace polynomial_roots_product_l2177_217788

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^6 - b*x - c = 0) → 
  b * c = 2030 := by
sorry

end polynomial_roots_product_l2177_217788


namespace sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l2177_217798

theorem sum_of_three_squares_divisible_by_three_to_not_divisible_by_three
  (N : ℕ) (a b c : ℤ) (h1 : ∃ (a b c : ℤ), N = a^2 + b^2 + c^2)
  (h2 : 3 ∣ a) (h3 : 3 ∣ b) (h4 : 3 ∣ c) :
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z) :=
by sorry

end sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l2177_217798


namespace family_ages_l2177_217757

/-- Represents the ages of a family with a father, mother, and three daughters. -/
structure FamilyAges where
  father : ℕ
  mother : ℕ
  eldest : ℕ
  middle : ℕ
  youngest : ℕ

/-- The family ages satisfy the given conditions. -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  -- Total age is 90
  ages.father + ages.mother + ages.eldest + ages.middle + ages.youngest = 90 ∧
  -- Age difference between daughters is 2 years
  ages.eldest = ages.middle + 2 ∧
  ages.middle = ages.youngest + 2 ∧
  -- Mother's age is 10 years more than sum of daughters' ages
  ages.mother = ages.eldest + ages.middle + ages.youngest + 10 ∧
  -- Age difference between father and mother equals middle daughter's age
  ages.father - ages.mother = ages.middle

/-- The theorem stating the ages of the family members. -/
theorem family_ages : ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
  ages.father = 38 ∧ ages.mother = 31 ∧ ages.eldest = 9 ∧ ages.middle = 7 ∧ ages.youngest = 5 := by
  sorry

end family_ages_l2177_217757


namespace expression_value_l2177_217797

theorem expression_value (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x^2 - 4 * y + 2 * z = 21 := by
  sorry

end expression_value_l2177_217797


namespace laptop_selection_theorem_l2177_217735

-- Define the number of laptops of each type
def typeA : ℕ := 4
def typeB : ℕ := 5

-- Define the total number of laptops to be selected
def selectTotal : ℕ := 3

-- Define the function to calculate the number of selections
def numSelections : ℕ := 
  Nat.choose typeA 2 * Nat.choose typeB 1 + 
  Nat.choose typeA 1 * Nat.choose typeB 2

-- Theorem statement
theorem laptop_selection_theorem : numSelections = 70 := by
  sorry

end laptop_selection_theorem_l2177_217735


namespace average_temperature_l2177_217763

/-- The average temperature of three days with recorded temperatures of -14°F, -8°F, and +1°F is -7°F. -/
theorem average_temperature (temp1 temp2 temp3 : ℚ) : 
  temp1 = -14 → temp2 = -8 → temp3 = 1 → (temp1 + temp2 + temp3) / 3 = -7 := by
  sorry

end average_temperature_l2177_217763


namespace angle_between_vectors_l2177_217707

/-- The angle between two vectors given their components and projection. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h : Real.sqrt 3 * (3 : ℝ) = (b.2 : ℝ)) 
  (proj : (3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2 = 3 * Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2)) :
  let angle := Real.arccos ((3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2) / 
    (Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2) * Real.sqrt ((3 : ℝ)^2 + b.2^2))
  angle = π / 6 :=
by sorry

end angle_between_vectors_l2177_217707


namespace contrapositive_equivalence_l2177_217743

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by sorry

end contrapositive_equivalence_l2177_217743


namespace sheet_width_is_36_l2177_217741

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure SheetAndBox where
  sheet_length : ℝ
  sheet_width : ℝ
  cut_square_side : ℝ
  box_volume : ℝ

/-- Calculates the volume of the box formed from the sheet. -/
def box_volume (s : SheetAndBox) : ℝ :=
  (s.sheet_length - 2 * s.cut_square_side) * (s.sheet_width - 2 * s.cut_square_side) * s.cut_square_side

/-- Theorem stating that given the specified conditions, the width of the sheet is 36 meters. -/
theorem sheet_width_is_36 (s : SheetAndBox) 
    (h1 : s.sheet_length = 48)
    (h2 : s.cut_square_side = 7)
    (h3 : s.box_volume = 5236)
    (h4 : box_volume s = s.box_volume) : 
  s.sheet_width = 36 := by
  sorry

#check sheet_width_is_36

end sheet_width_is_36_l2177_217741


namespace sequence_divisibility_l2177_217778

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => (sequence_a n)^2 + sequence_a n + 1

theorem sequence_divisibility (n : ℕ) : 
  (n ≥ 1) → (sequence_a n)^2 + 1 ∣ (sequence_a (n + 1))^2 + 1 := by
  sorry

end sequence_divisibility_l2177_217778


namespace max_a_value_l2177_217736

/-- A function f defined on the positive reals satisfying certain properties -/
def f : ℝ → ℝ :=
  sorry

/-- The conditions on f -/
axiom f_add (x y : ℝ) : x > 0 → y > 0 → f x + f y = f (x * y)

axiom f_neg (x : ℝ) : x > 1 → f x < 0

axiom f_ineq (x y a : ℝ) : x > 0 → y > 0 → a > 0 → 
  f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))) →
  a ≤ Real.sqrt 2 :=
sorry

end max_a_value_l2177_217736


namespace promotions_equivalent_l2177_217781

/-- Calculates the discount percentage for a given promotion --/
def discount_percentage (items_taken : ℕ) (items_paid : ℕ) : ℚ :=
  (items_taken - items_paid : ℚ) / items_taken * 100

/-- The original promotion "Buy one and get another for half price" --/
def original_promotion : ℚ := discount_percentage 2 (3/2)

/-- The alternative promotion "Take four and pay for three" --/
def alternative_promotion : ℚ := discount_percentage 4 3

/-- Theorem stating that both promotions offer the same discount --/
theorem promotions_equivalent : original_promotion = alternative_promotion := by
  sorry

end promotions_equivalent_l2177_217781


namespace original_serving_size_l2177_217737

/-- Proves that the original serving size was 8 ounces -/
theorem original_serving_size (total_water : ℝ) (current_serving : ℝ) (serving_difference : ℕ) : 
  total_water = 64 →
  current_serving = 16 →
  (total_water / current_serving : ℝ) + serving_difference = total_water / 8 →
  8 = total_water / ((total_water / current_serving : ℝ) + serving_difference : ℝ) := by
sorry

end original_serving_size_l2177_217737


namespace equal_roots_quadratic_l2177_217792

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 4 = 0 → y = x) → 
  a = 4 := by
sorry

end equal_roots_quadratic_l2177_217792


namespace hyperbola_line_inclination_l2177_217714

/-- Given a hyperbola with equation x²/m² - y²/n² = 1 and eccentricity 2,
    prove that the angle of inclination of the line mx + ny - 1 = 0
    is either π/6 or 5π/6 -/
theorem hyperbola_line_inclination (m n : ℝ) (h_eccentricity : m^2 + n^2 = 4 * m^2) :
  let θ := Real.arctan (-m / n)
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end hyperbola_line_inclination_l2177_217714


namespace friend_payment_percentage_l2177_217799

def adoption_fee : ℝ := 200
def james_payment : ℝ := 150

theorem friend_payment_percentage : 
  (adoption_fee - james_payment) / adoption_fee * 100 = 25 := by sorry

end friend_payment_percentage_l2177_217799


namespace factorization_of_cubic_l2177_217761

theorem factorization_of_cubic (x : ℝ) : 3 * x^3 - 27 * x = 3 * x * (x + 3) * (x - 3) := by
  sorry

end factorization_of_cubic_l2177_217761


namespace exists_range_sum_and_even_count_611_l2177_217771

/-- Sum of integers from a to b (inclusive) -/
def sum_range (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Count of even integers in range [a, b] -/
def count_even (a b : ℤ) : ℤ :=
  if a % 2 = 0 && b % 2 = 0 then
    (b - a) / 2 + 1
  else
    (b - a + 1) / 2

theorem exists_range_sum_and_even_count_611 :
  ∃ a b : ℤ, sum_range a b + count_even a b = 611 :=
sorry

end exists_range_sum_and_even_count_611_l2177_217771


namespace container_fullness_l2177_217710

theorem container_fullness 
  (capacity : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : capacity = 120)
  (h2 : initial_percentage = 0.3)
  (h3 : added_water = 54) :
  (initial_percentage * capacity + added_water) / capacity = 0.75 :=
by sorry

end container_fullness_l2177_217710


namespace consecutive_integers_square_sum_l2177_217738

theorem consecutive_integers_square_sum (x : ℤ) : 
  x^2 + (x+1)^2 + x^2 * (x+1)^2 = (x^2 + x + 1)^2 := by
  sorry

end consecutive_integers_square_sum_l2177_217738


namespace simplify_expression_l2177_217704

theorem simplify_expression (r s : ℝ) : 120*r - 32*r + 50*s - 20*s = 88*r + 30*s := by
  sorry

end simplify_expression_l2177_217704


namespace balls_in_boxes_l2177_217789

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

-- State the theorem
theorem balls_in_boxes : 
  distribute_balls num_balls num_boxes = 21 := by
  sorry

end balls_in_boxes_l2177_217789


namespace isosceles_triangle_with_60_degree_angle_l2177_217749

theorem isosceles_triangle_with_60_degree_angle (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧ -- Angles are positive
  α + β + β = 180 ∧ -- Sum of angles in a triangle
  α = 60 ∧ -- One angle is 60°
  β = β -- Triangle is isosceles with two equal angles
  → α = 60 ∧ β = 60 := by sorry

end isosceles_triangle_with_60_degree_angle_l2177_217749


namespace initial_fish_count_l2177_217756

/-- The number of fish moved to a different tank -/
def fish_moved : ℕ := 68

/-- The number of fish remaining in the first tank -/
def fish_remaining : ℕ := 144

/-- The initial number of fish in the first tank -/
def initial_fish : ℕ := fish_moved + fish_remaining

theorem initial_fish_count : initial_fish = 212 := by
  sorry

end initial_fish_count_l2177_217756


namespace root_sum_equals_three_l2177_217787

theorem root_sum_equals_three (x₁ x₂ : ℝ) 
  (h₁ : x₁ + Real.log x₁ = 3) 
  (h₂ : x₂ + (10 : ℝ) ^ x₂ = 3) : 
  x₁ + x₂ = 3 := by sorry

end root_sum_equals_three_l2177_217787


namespace inequality_implication_l2177_217777

theorem inequality_implication (m a b : ℝ) : a * m^2 > b * m^2 → a > b := by
  sorry

end inequality_implication_l2177_217777


namespace crow_worm_consumption_l2177_217746

theorem crow_worm_consumption 
  (crows_per_hour : ℕ) 
  (worms_per_hour : ℕ) 
  (new_crows : ℕ) 
  (new_hours : ℕ) 
  (h1 : crows_per_hour = 3) 
  (h2 : worms_per_hour = 30) 
  (h3 : new_crows = 5) 
  (h4 : new_hours = 2) : 
  (worms_per_hour / crows_per_hour) * new_crows * new_hours = 100 := by
  sorry

end crow_worm_consumption_l2177_217746


namespace binomial_expectation_six_half_l2177_217759

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/2) is 3 -/
theorem binomial_expectation_six_half :
  let X : BinomialDistribution := ⟨6, 1/2, by norm_num⟩
  expectation X = 3 := by sorry

end binomial_expectation_six_half_l2177_217759


namespace arithmetic_sequence_common_difference_l2177_217732

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l2177_217732


namespace encryption_decryption_l2177_217752

/-- Given an encryption formula y = a^x - 2, prove that when a^3 - 2 = 6 and y = 14, x = 4 --/
theorem encryption_decryption (a : ℝ) (h1 : a^3 - 2 = 6) (y : ℝ) (h2 : y = 14) :
  ∃ x : ℝ, a^x - 2 = y ∧ x = 4 := by
  sorry

end encryption_decryption_l2177_217752


namespace fencing_cost_for_specific_plot_l2177_217742

/-- Represents a rectangular plot with its dimensions in meters -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Calculates the total cost of fencing a plot given the cost per meter -/
def fencingCost (plot : RectangularPlot) (costPerMeter : ℝ) : ℝ :=
  costPerMeter * perimeter plot

/-- Theorem stating the total cost of fencing for a specific rectangular plot -/
theorem fencing_cost_for_specific_plot :
  let plot : RectangularPlot := { length := 60, breadth := 40 }
  let costPerMeter : ℝ := 26.5
  fencingCost plot costPerMeter = 5300 := by
  sorry

#check fencing_cost_for_specific_plot

end fencing_cost_for_specific_plot_l2177_217742


namespace regular_polygon_with_150_degree_angles_l2177_217747

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 := by
  sorry

end regular_polygon_with_150_degree_angles_l2177_217747


namespace inequality_proof_l2177_217772

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end inequality_proof_l2177_217772


namespace expression_value_l2177_217760

theorem expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 1)*(x - 1) = 3 := by
  sorry

end expression_value_l2177_217760


namespace max_earnings_is_250_l2177_217719

/-- Represents a plumbing job with counts of toilets, showers, and sinks to be fixed -/
structure PlumbingJob where
  toilets : ℕ
  showers : ℕ
  sinks : ℕ

/-- Calculates the earnings for a given plumbing job -/
def jobEarnings (job : PlumbingJob) : ℕ :=
  job.toilets * 50 + job.showers * 40 + job.sinks * 30

/-- The list of available jobs -/
def availableJobs : List PlumbingJob := [
  { toilets := 3, showers := 0, sinks := 3 },
  { toilets := 2, showers := 0, sinks := 5 },
  { toilets := 1, showers := 2, sinks := 3 }
]

/-- Theorem stating that the maximum earnings from the available jobs is $250 -/
theorem max_earnings_is_250 : 
  (availableJobs.map jobEarnings).maximum? = some 250 := by sorry

end max_earnings_is_250_l2177_217719


namespace min_perimeter_nine_square_rectangle_l2177_217733

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ  -- Side length of the smallest square
  b : ℕ  -- Side length of the second smallest square
  length : ℕ  -- Length of the rectangle
  width : ℕ   -- Width of the rectangle

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- Conditions for a valid NineSquareRectangle configuration -/
def is_valid_configuration (rect : NineSquareRectangle) : Prop :=
  rect.b = 3 * rect.a ∧
  rect.length = 2 * rect.a + rect.b + 3 * rect.a + rect.b ∧
  rect.width = 12 * rect.a - 2 * rect.b + 8 * rect.a - rect.b

/-- Theorem stating the smallest possible perimeter of a NineSquareRectangle -/
theorem min_perimeter_nine_square_rectangle :
  ∀ rect : NineSquareRectangle, is_valid_configuration rect →
  perimeter rect ≥ 52 :=
sorry

end min_perimeter_nine_square_rectangle_l2177_217733


namespace remainder_theorem_remainder_is_72_l2177_217708

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 8*x - 20

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + a) * q x + f (-a) :=
sorry

theorem remainder_is_72 : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + 2) * q x + 72 :=
sorry

end remainder_theorem_remainder_is_72_l2177_217708


namespace megan_folders_l2177_217715

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  files_per_folder = 8 →
  (initial_files - deleted_files) / files_per_folder = 9 :=
by sorry

end megan_folders_l2177_217715


namespace max_value_theorem_l2177_217720

theorem max_value_theorem (x y : ℝ) (h : x^2 - 3*x + 4*y = 7) :
  ∃ (M : ℝ), M = 16 ∧ ∀ (x' y' : ℝ), x'^2 - 3*x' + 4*y' = 7 → 3*x' + 4*y' ≤ M :=
sorry

end max_value_theorem_l2177_217720


namespace monotonicity_condition_l2177_217727

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the monotonicity property on the interval (-∞, 8]
def is_monotonic_on_interval (k : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 8 → f k x < f k y ∨ f k x > f k y

-- Theorem statement
theorem monotonicity_condition (k : ℝ) :
  is_monotonic_on_interval k ↔ k ≥ 64 := by
  sorry

end monotonicity_condition_l2177_217727


namespace clock_hands_minimum_time_l2177_217768

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60
    minutes := m % 60
    valid := by sorry }

theorem clock_hands_minimum_time :
  let t1 : Time := { hours := 0, minutes := 45, valid := by sorry }
  let t2 : Time := { hours := 3, minutes := 30, valid := by sorry }
  let diff := timeDifferenceInMinutes t1 t2
  let result := minutesToTime diff
  result.hours = 2 ∧ result.minutes = 45 := by sorry

end clock_hands_minimum_time_l2177_217768


namespace eunji_class_size_l2177_217725

/-- The number of lines students stand in --/
def num_lines : ℕ := 3

/-- Eunji's position from the front of the line --/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of the line --/
def position_from_back : ℕ := 6

/-- The total number of students in Eunji's line --/
def students_per_line : ℕ := position_from_front + position_from_back - 1

/-- The total number of students in Eunji's class --/
def total_students : ℕ := num_lines * students_per_line

theorem eunji_class_size : total_students = 24 := by
  sorry

end eunji_class_size_l2177_217725


namespace magnitude_of_z_l2177_217718

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + i) * z = 3 + i

-- State the theorem
theorem magnitude_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_z_l2177_217718


namespace line_symmetry_l2177_217726

-- Define the lines
def line_l (x y : ℝ) : Prop := x - y - 1 = 0
def line_l1 (x y : ℝ) : Prop := 2*x - y - 2 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ ∃ (x' y' : ℝ), l2 x' y' ∧ l ((x + x')/2) ((y + y')/2)

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt line_l1 line_l2 line_l :=
sorry

end line_symmetry_l2177_217726


namespace bert_spent_nine_at_dry_cleaners_l2177_217765

/-- Represents Bert's spending problem -/
def BertSpending (initial_amount dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store_amount := initial_amount / 4
  let after_hardware := initial_amount - hardware_store_amount
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store_amount := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store_amount
  (initial_amount = 52) ∧
  (final_amount = 15) ∧
  (dry_cleaner_amount > 0)

/-- Proves that Bert spent $9 at the dry cleaners -/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 52 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end bert_spent_nine_at_dry_cleaners_l2177_217765


namespace intersection_minimizes_sum_of_distances_l2177_217790

/-- Given a triangle ABC, construct equilateral triangles ABC₁, ACB₁, and BCA₁ externally --/
def constructExternalTriangles (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Compute the intersection point of lines AA₁, BB₁, and CC₁ --/
def intersectionPoint (A B C : ℝ × ℝ) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Compute the sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (P A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The intersection point minimizes the sum of distances --/
theorem intersection_minimizes_sum_of_distances (A B C : ℝ × ℝ) :
  let (A₁, B₁, C₁) := constructExternalTriangles A B C
  let O := intersectionPoint A B C A₁ B₁ C₁
  ∀ P : ℝ × ℝ, sumOfDistances O A B C ≤ sumOfDistances P A B C :=
sorry

end intersection_minimizes_sum_of_distances_l2177_217790


namespace complex_expression_evaluation_l2177_217750

theorem complex_expression_evaluation : 
  (39/7) / ((8.4 * (6/7) * (6 - ((2.3 + 5/6.25) * 7) / (8 * 0.0125 + 6.9))) - 20.384/1.3) = 15/14 := by
  sorry

end complex_expression_evaluation_l2177_217750


namespace equation_satisfaction_l2177_217762

theorem equation_satisfaction (a b c : ℤ) (h1 : a = c) (h2 : b + 1 = c) :
  a * (b - c) + b * (c - a) + c * (a - b) = -1 := by
  sorry

end equation_satisfaction_l2177_217762


namespace fraction_addition_l2177_217795

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 9 = (11 : ℚ) / 15 := by
  sorry

end fraction_addition_l2177_217795


namespace teachers_not_adjacent_arrangements_l2177_217767

def num_students : ℕ := 3
def num_teachers : ℕ := 2

def arrangement_count : ℕ := 72

theorem teachers_not_adjacent_arrangements :
  (Nat.factorial num_students) * (num_students + 1) * num_teachers = arrangement_count :=
by sorry

end teachers_not_adjacent_arrangements_l2177_217767


namespace train_length_proof_l2177_217729

theorem train_length_proof (passing_time : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  passing_time = 8 →
  platform_length = 279 →
  crossing_time = 20 →
  ∃ (train_length : ℝ),
    train_length = passing_time * (train_length + platform_length) / crossing_time ∧
    train_length = 186 :=
by sorry

end train_length_proof_l2177_217729


namespace smallest_positive_solution_l2177_217753

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ x/4 + 3/(4*x) = 1 ∧ ∀ (y : ℝ), y > 0 ∧ y/4 + 3/(4*y) = 1 → x ≤ y :=
by sorry

end smallest_positive_solution_l2177_217753


namespace sams_coins_value_l2177_217740

/-- Represents the value of Sam's coins in dollars -/
def total_value : ℚ :=
  let total_coins : ℕ := 30
  let nickels : ℕ := 12
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  (nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value

theorem sams_coins_value : total_value = 2.40 := by
  sorry

end sams_coins_value_l2177_217740


namespace max_distance_after_braking_l2177_217786

/-- The distance function for a car after braking -/
def s (b : ℝ) (t : ℝ) : ℝ := -6 * t^2 + b * t

/-- Theorem: Maximum distance traveled by the car after braking -/
theorem max_distance_after_braking (b : ℝ) :
  s b (1/2) = 6 → ∃ (t_max : ℝ), ∀ (t : ℝ), s b t ≤ s b t_max ∧ s b t_max = 75/8 := by
  sorry

end max_distance_after_braking_l2177_217786


namespace ratio_problem_l2177_217748

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 5) :
  x / y = 13 / 9 := by sorry

end ratio_problem_l2177_217748
