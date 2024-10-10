import Mathlib

namespace min_value_inequality_l1749_174960

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) : 
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end min_value_inequality_l1749_174960


namespace original_stones_count_l1749_174985

/-- The number of stones sent away to the Geological Museum in London. -/
def stones_sent_away : ℕ := 63

/-- The number of stones kept in the collection. -/
def stones_kept : ℕ := 15

/-- The original number of stones in the collection. -/
def original_stones : ℕ := stones_sent_away + stones_kept

/-- Theorem stating that the original number of stones in the collection is 78. -/
theorem original_stones_count : original_stones = 78 := by
  sorry

end original_stones_count_l1749_174985


namespace property_price_reduction_l1749_174983

theorem property_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 5000)
  (h2 : final_price = 4050)
  (h3 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.1 ∧ initial_price * (1 - x)^2 = final_price :=
by sorry

end property_price_reduction_l1749_174983


namespace S_intersect_T_eq_T_l1749_174973

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l1749_174973


namespace march_and_may_greatest_drop_l1749_174900

/-- Represents the months of the year --/
inductive Month
| January | February | March | April | May | June | July | August

/-- Price change for each month --/
def price_change : Month → ℝ
| Month.January  => -1.00
| Month.February => 1.50
| Month.March    => -3.00
| Month.April    => 2.00
| Month.May      => -3.00
| Month.June     => 0.50
| Month.July     => -2.50
| Month.August   => -1.50

/-- Predicate to check if a month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ n : Month, price_change m ≤ price_change n

/-- Theorem stating that March and May have the greatest monthly drop in price --/
theorem march_and_may_greatest_drop :
  has_greatest_drop Month.March ∧ has_greatest_drop Month.May :=
sorry

end march_and_may_greatest_drop_l1749_174900


namespace expression_simplification_l1749_174944

theorem expression_simplification (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let a : ℝ := 0.04
  1.24 * (Real.sqrt ((a * b * c + 4) / a + 4 * Real.sqrt (b * c / a))) / (Real.sqrt (a * b * c) + 2) = 6.2 := by
  sorry

end expression_simplification_l1749_174944


namespace performance_arrangements_l1749_174978

-- Define the number of singers
def n : ℕ := 6

-- Define the number of singers with specific order requirements (A, B, C)
def k : ℕ := 3

-- Define the number of valid orders for B and C relative to A
def valid_orders : ℕ := 4

-- Theorem statement
theorem performance_arrangements : 
  (valid_orders : ℕ) * (n.factorial / k.factorial) = 480 := by
  sorry

end performance_arrangements_l1749_174978


namespace smallest_n_divisible_l1749_174981

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 864 ∣ m^3)) ∧ 
  (24 ∣ n^2) ∧ (864 ∣ n^3) → n = 12 := by
  sorry

end smallest_n_divisible_l1749_174981


namespace complex_fraction_problem_l1749_174940

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59405 / 30958 := by
  sorry

end complex_fraction_problem_l1749_174940


namespace intersection_of_lines_l1749_174963

theorem intersection_of_lines : 
  ∃! (x y : ℚ), 2 * y = 3 * x ∧ 3 * y + 1 = -6 * x ∧ x = -2/21 ∧ y = -1/7 :=
by sorry

end intersection_of_lines_l1749_174963


namespace reflected_triangle_angles_l1749_174951

-- Define the triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the reflection operation
def reflect (t : Triangle) : Triangle := sorry

-- Define the property of being acute
def is_acute (t : Triangle) : Prop := sorry

-- Define the property of being scalene
def is_scalene (t : Triangle) : Prop := sorry

-- Define the property of A₁B₁C₁ being inside ABC
def is_inside (t1 t2 : Triangle) : Prop := sorry

-- Define the property of A₂B₂C₂ being outside ABC
def is_outside (t1 t2 : Triangle) : Prop := sorry

-- Define the theorem
theorem reflected_triangle_angles 
  (ABC : Triangle) 
  (A₁B₁C₁ : Triangle) 
  (A₂B₂C₂ : Triangle) :
  is_acute ABC →
  is_scalene ABC →
  is_inside A₁B₁C₁ ABC →
  is_outside A₂B₂C₂ ABC →
  (A₂B₂C₂.a = 20 ∨ A₂B₂C₂.b = 20 ∨ A₂B₂C₂.c = 20) →
  (A₂B₂C₂.a = 70 ∨ A₂B₂C₂.b = 70 ∨ A₂B₂C₂.c = 70) →
  ((A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 60) ∨
   (A₁B₁C₁.a = 140/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 140/3 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 220/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 140/3)) :=
by sorry

end reflected_triangle_angles_l1749_174951


namespace probability_three_and_zero_painted_faces_l1749_174935

/-- Represents a 5x5x5 cube with three faces sharing a common corner painted red -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Counts the number of unit cubes with a specific number of painted faces -/
def count_painted_cubes (c : PaintedCube) (num_painted_faces : ℕ) : ℕ :=
  sorry

/-- Calculates the probability of selecting two specific types of unit cubes -/
def probability_of_selection (c : PaintedCube) (faces1 faces2 : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of selecting one cube with three
    painted faces and one cube with no painted faces -/
theorem probability_three_and_zero_painted_faces (c : PaintedCube) :
  c.side_length = 5 ∧ c.total_cubes = 125 ∧ c.painted_faces = 3 →
  probability_of_selection c 3 0 = 44 / 3875 :=
sorry

end probability_three_and_zero_painted_faces_l1749_174935


namespace ones_digit_of_33_power_l1749_174924

theorem ones_digit_of_33_power (n : ℕ) : 
  (33^(33*(12^12))) % 10 = 1 := by
sorry

end ones_digit_of_33_power_l1749_174924


namespace mixed_yellow_ratio_is_quarter_l1749_174906

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Calculates the ratio of yellow jelly beans to total jelly beans when multiple bags are mixed -/
def mixed_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_yellow := bags.map yellow_count |>.sum
  let total_beans := bags.map (·.total) |>.sum
  total_yellow / total_beans

theorem mixed_yellow_ratio_is_quarter (bags : List JellyBeanBag) :
  bags = [
    ⟨24, 2/5⟩,
    ⟨30, 3/10⟩,
    ⟨32, 1/4⟩,
    ⟨34, 1/10⟩
  ] →
  mixed_yellow_ratio bags = 1/4 := by
  sorry

end mixed_yellow_ratio_is_quarter_l1749_174906


namespace exponential_function_fixed_point_l1749_174948

/-- The function f(x) = a^(-x-2) + 4 always passes through the point (-2, 5) for a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(-x-2) + 4
  f (-2) = 5 := by
sorry

end exponential_function_fixed_point_l1749_174948


namespace min_value_w_l1749_174956

theorem min_value_w (x y : ℝ) : 
  ∃ (w_min : ℝ), w_min = 20.25 ∧ ∀ (w : ℝ), w = 3*x^2 + 3*y^2 + 9*x - 6*y + 27 → w ≥ w_min :=
by sorry

end min_value_w_l1749_174956


namespace gathering_attendance_l1749_174953

theorem gathering_attendance (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 := by sorry

end gathering_attendance_l1749_174953


namespace gcd_1681_1705_l1749_174932

theorem gcd_1681_1705 : Nat.gcd 1681 1705 = 1 := by
  sorry

end gcd_1681_1705_l1749_174932


namespace transfer_increases_averages_l1749_174987

/-- Represents a group of students with their total score and count -/
structure StudentGroup where
  totalScore : ℚ
  count : ℕ

/-- Calculates the average score of a group -/
def averageScore (group : StudentGroup) : ℚ :=
  group.totalScore / group.count

/-- Theorem: Transferring Lopatin and Filin increases average scores in both groups -/
theorem transfer_increases_averages
  (groupA groupB : StudentGroup)
  (lopatinScore filinScore : ℚ)
  (h1 : groupA.count = 10)
  (h2 : groupB.count = 10)
  (h3 : averageScore groupA = 47.2)
  (h4 : averageScore groupB = 41.8)
  (h5 : 41.8 < lopatinScore) (h6 : lopatinScore < 47.2)
  (h7 : 41.8 < filinScore) (h8 : filinScore < 47.2)
  (h9 : lopatinScore = 47)
  (h10 : filinScore = 44) :
  let newGroupA : StudentGroup := ⟨groupA.totalScore - lopatinScore - filinScore, 8⟩
  let newGroupB : StudentGroup := ⟨groupB.totalScore + lopatinScore + filinScore, 12⟩
  averageScore newGroupA > 47.5 ∧ averageScore newGroupB > 42.2 := by
  sorry

end transfer_increases_averages_l1749_174987


namespace square_difference_l1749_174993

theorem square_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 := by
sorry

end square_difference_l1749_174993


namespace fraction_simplification_l1749_174980

theorem fraction_simplification (a b c : ℝ) 
  (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := by
  sorry

end fraction_simplification_l1749_174980


namespace distance_calculation_l1749_174904

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 54

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 6

/-- The time Brad runs before meeting Maxwell, in hours -/
def brad_time : ℝ := maxwell_time - 1

theorem distance_calculation :
  distance_between_homes = maxwell_speed * maxwell_time + brad_speed * brad_time :=
by sorry

end distance_calculation_l1749_174904


namespace rectangular_field_dimensions_l1749_174914

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  width_length_diff : ℕ

/-- Checks if the given length and width satisfy the conditions for a rectangular field -/
def is_valid_dimensions (field : RectangularField) (length width : ℕ) : Prop :=
  length * width = field.area ∧ length = width + field.width_length_diff

theorem rectangular_field_dimensions (field : RectangularField) 
  (h : field.area = 864 ∧ field.width_length_diff = 12) :
  ∃ (length width : ℕ), is_valid_dimensions field length width ∧ length = 36 ∧ width = 24 := by
  sorry

end rectangular_field_dimensions_l1749_174914


namespace initial_salt_percentage_l1749_174920

theorem initial_salt_percentage
  (initial_mass : ℝ)
  (added_salt : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_mass = 100)
  (h2 : added_salt = 38.46153846153846)
  (h3 : final_percentage = 35) :
  let final_mass := initial_mass + added_salt
  let final_salt_mass := (final_percentage / 100) * final_mass
  let initial_salt_mass := final_salt_mass - added_salt
  initial_salt_mass / initial_mass * 100 = 10 := by
sorry

end initial_salt_percentage_l1749_174920


namespace system_solution_l1749_174957

variables (a b x y : ℝ)

theorem system_solution (h1 : x / (a - 2*b) - y / (a + 2*b) = (6*a*b) / (a^2 - 4*b^2))
                        (h2 : (x + y) / (a + 2*b) + (x - y) / (a - 2*b) = (2*(a^2 - a*b + 2*b^2)) / (a^2 - 4*b^2))
                        (h3 : a ≠ 2*b)
                        (h4 : a ≠ -2*b)
                        (h5 : a^2 ≠ 4*b^2) :
  x = a + b ∧ y = a - b :=
by sorry


end system_solution_l1749_174957


namespace rectangle_area_diagonal_ratio_l1749_174919

theorem rectangle_area_diagonal_ratio (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 4 / 3 →
  diagonal ^ 2 = length ^ 2 + width ^ 2 →
  length * width = k * diagonal ^ 2 →
  k = 12 / 25 := by
sorry

end rectangle_area_diagonal_ratio_l1749_174919


namespace fraction_meaningful_l1749_174931

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by sorry

end fraction_meaningful_l1749_174931


namespace number_ordering_l1749_174905

theorem number_ordering : 7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 := by
  sorry

end number_ordering_l1749_174905


namespace unique_solution_in_p_arithmetic_l1749_174937

-- Define p-arithmetic structure
structure PArithmetic (p : ℕ) where
  carrier : Type
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  zero : carrier
  one : carrier
  -- Add necessary axioms for p-arithmetic

-- Define the theorem
theorem unique_solution_in_p_arithmetic {p : ℕ} (P : PArithmetic p) :
  ∀ (a b : P.carrier), a ≠ P.zero → ∃! x : P.carrier, P.mul a x = b :=
sorry

end unique_solution_in_p_arithmetic_l1749_174937


namespace crabapple_theorem_l1749_174936

/-- The number of different sequences of crabapple recipients in a week for two classes -/
def crabapple_sequences (students1 : ℕ) (meetings1 : ℕ) (students2 : ℕ) (meetings2 : ℕ) : ℕ :=
  (students1 ^ meetings1) * (students2 ^ meetings2)

/-- Theorem stating the number of crabapple recipient sequences for the given classes -/
theorem crabapple_theorem :
  crabapple_sequences 12 3 9 2 = 139968 := by
  sorry

#eval crabapple_sequences 12 3 9 2

end crabapple_theorem_l1749_174936


namespace room_length_proof_l1749_174946

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 6 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 25650 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 6 := by
sorry

end room_length_proof_l1749_174946


namespace twelfth_term_of_sequence_l1749_174996

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

-- State the theorem
theorem twelfth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/4) :
  arithmetic_sequence a₁ d 12 = 3 :=
by
  sorry

end twelfth_term_of_sequence_l1749_174996


namespace rectangle_length_is_16_l1749_174974

/-- Proves that the length of a rectangle is 16 cm given specific conditions --/
theorem rectangle_length_is_16 (b : ℝ) (c : ℝ) :
  b = 14 →
  c = 23.56 →
  ∃ (l : ℝ), l = 16 ∧ 
    2 * (l + b) = 4 * (c / π) ∧
    c / π = (2 * c) / (2 * π) :=
by sorry

end rectangle_length_is_16_l1749_174974


namespace geometric_sequence_sum_l1749_174988

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 2 = 3) : 
  a 3 + a 4 = 12 := by
  sorry

end geometric_sequence_sum_l1749_174988


namespace parabola_translation_l1749_174955

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let translated := translate original 3 4
  y = translated.a * x^2 + translated.b * x + translated.c ↔
  y = 2 * (x + 3)^2 + 4 := by
  sorry

end parabola_translation_l1749_174955


namespace arcsin_arccos_half_pi_l1749_174949

theorem arcsin_arccos_half_pi : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (1/2) = π/3 := by sorry

end arcsin_arccos_half_pi_l1749_174949


namespace five_ps_high_gpa_l1749_174990

/-- Represents the number of applicants satisfying various criteria in a law school application process. -/
structure Applicants where
  total : ℕ
  political_science : ℕ
  high_gpa : ℕ
  not_ps_low_gpa : ℕ

/-- Calculates the number of applicants who majored in political science and had a GPA higher than 3.0. -/
def political_science_and_high_gpa (a : Applicants) : ℕ :=
  a.high_gpa - (a.total - a.political_science - a.not_ps_low_gpa)

/-- Theorem stating that for the given applicant data, 5 applicants majored in political science and had a GPA higher than 3.0. -/
theorem five_ps_high_gpa (a : Applicants) 
    (h_total : a.total = 40)
    (h_ps : a.political_science = 15)
    (h_high_gpa : a.high_gpa = 20)
    (h_not_ps_low_gpa : a.not_ps_low_gpa = 10) :
    political_science_and_high_gpa a = 5 := by
  sorry

#eval political_science_and_high_gpa ⟨40, 15, 20, 10⟩

end five_ps_high_gpa_l1749_174990


namespace smaller_square_area_l1749_174907

theorem smaller_square_area (larger_square_area : ℝ) 
  (h1 : larger_square_area = 144) 
  (h2 : ∀ (side : ℝ), side * side = larger_square_area → 
        ∃ (smaller_side : ℝ), smaller_side = side / 2) : 
  ∃ (smaller_area : ℝ), smaller_area = 72 := by
sorry

end smaller_square_area_l1749_174907


namespace tangent_line_triangle_area_l1749_174917

-- Define the function f(x) = x³ - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- State the theorem
theorem tangent_line_triangle_area :
  let tangent_slope : ℝ := f' 0
  let tangent_y_intercept : ℝ := 1
  let tangent_x_intercept : ℝ := 1
  (1 / 2) * tangent_x_intercept * tangent_y_intercept = 1 / 2 :=
by sorry

end tangent_line_triangle_area_l1749_174917


namespace odometer_problem_l1749_174958

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c ≤ 10) 
  (hdiv : ∃ t : ℕ, (100 * a + 10 * c) - (100 * a + 10 * b + c) = 60 * t) :
  a^2 + b^2 + c^2 = 26 := by
sorry

end odometer_problem_l1749_174958


namespace cargo_transport_possible_l1749_174964

/-- Represents the cargo transportation problem -/
structure CargoTransport where
  totalCargo : ℕ
  bagCapacity : ℕ
  truckCapacity : ℕ
  maxTrips : ℕ

/-- Checks if the cargo can be transported within the given number of trips -/
def canTransport (ct : CargoTransport) : Prop :=
  ∃ (trips : ℕ), trips ≤ ct.maxTrips ∧ trips * ct.truckCapacity ≥ ct.totalCargo

/-- Theorem stating that 36 tons of cargo can be transported in 11 trips or fewer -/
theorem cargo_transport_possible : 
  canTransport ⟨36, 1, 4, 11⟩ := by
  sorry

#check cargo_transport_possible

end cargo_transport_possible_l1749_174964


namespace annas_gold_amount_annas_gold_theorem_l1749_174928

theorem annas_gold_amount (gary_gold : ℕ) (gary_cost_per_gram : ℕ) (anna_cost_per_gram : ℕ) (total_cost : ℕ) : ℕ :=
  let gary_total_cost := gary_gold * gary_cost_per_gram
  let anna_total_cost := total_cost - gary_total_cost
  anna_total_cost / anna_cost_per_gram

theorem annas_gold_theorem :
  annas_gold_amount 30 15 20 1450 = 50 := by
  sorry

end annas_gold_amount_annas_gold_theorem_l1749_174928


namespace runners_in_picture_probability_l1749_174921

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of two runners being in a picture -/
def probabilityInPicture (jenny : Runner) (jack : Runner) : ℚ :=
  -- Define the probability calculation here
  23 / 60

/-- Theorem stating the probability of Jenny and Jack being in the picture -/
theorem runners_in_picture_probability :
  let jenny : Runner := { lapTime := 75, direction := true }
  let jack : Runner := { lapTime := 70, direction := false }
  let pictureTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let pictureDuration : ℝ := 60  -- 1 minute in seconds
  let pictureTrackCoverage : ℝ := 1 / 3
  probabilityInPicture jenny jack = 23 / 60 := by
  sorry

#eval probabilityInPicture { lapTime := 75, direction := true } { lapTime := 70, direction := false }

end runners_in_picture_probability_l1749_174921


namespace marias_car_trip_l1749_174916

theorem marias_car_trip (D : ℝ) : 
  D / 2 + (D - D / 2) / 4 + 210 = D → D = 560 := by sorry

end marias_car_trip_l1749_174916


namespace PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l1749_174947

/-- Proposition A: x ≠ 2 or y ≠ 3 -/
def PropA (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 3

/-- Proposition B: x + y ≠ 5 -/
def PropB (x y : ℝ) : Prop := x + y ≠ 5

/-- Proposition B implies Proposition A -/
theorem PropB_implies_PropA : ∀ x y : ℝ, PropB x y → PropA x y := by sorry

/-- Proposition A does not imply Proposition B -/
theorem PropA_not_implies_PropB : ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

/-- A is a necessary but not sufficient condition for B -/
theorem A_necessary_not_sufficient_for_B : 
  (∀ x y : ℝ, PropB x y → PropA x y) ∧ ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

end PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l1749_174947


namespace units_digit_product_minus_cube_l1749_174952

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_product_minus_cube : units_digit (8 * 18 * 1998 - 8^3) = 0 := by
  sorry

end units_digit_product_minus_cube_l1749_174952


namespace one_third_in_one_sixth_l1749_174986

theorem one_third_in_one_sixth :
  (1 : ℚ) / 6 / ((1 : ℚ) / 3) = 1 / 2 := by
sorry

end one_third_in_one_sixth_l1749_174986


namespace cube_volume_from_surface_area_l1749_174910

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 24 → volume = 8 := by
  sorry

end cube_volume_from_surface_area_l1749_174910


namespace two_digit_reverse_sum_l1749_174908

/-- Two-digit integer -/
def TwoDigitInt (z : ℕ) : Prop := 10 ≤ z ∧ z ≤ 99

/-- Reverse digits of a two-digit number -/
def reverseDigits (x : ℕ) : ℕ := 10 * (x % 10) + (x / 10)

theorem two_digit_reverse_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 - y^2 = n^2 → x + y + n = 154 := by
  sorry

end two_digit_reverse_sum_l1749_174908


namespace rhombus_longer_diagonal_l1749_174971

/-- A rhombus with side length 40 units and shorter diagonal 56 units has a longer diagonal of length 24√17 units. -/
theorem rhombus_longer_diagonal (s : ℝ) (d₁ : ℝ) (d₂ : ℝ) 
    (h₁ : s = 40) 
    (h₂ : d₁ = 56) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : 
  d₂ = 24 * Real.sqrt 17 := by sorry

end rhombus_longer_diagonal_l1749_174971


namespace sum_bounds_l1749_174918

theorem sum_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end sum_bounds_l1749_174918


namespace solve_system_l1749_174998

theorem solve_system (x y : ℚ) (h1 : x - y = 12) (h2 : 2 * x + y = 10) : y = -14/3 := by
  sorry

end solve_system_l1749_174998


namespace gcd_diff_is_square_l1749_174942

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x (Nat.gcd y z)) * (y - x) = k ^ 2 := by
  sorry

end gcd_diff_is_square_l1749_174942


namespace division_problem_l1749_174991

theorem division_problem (n : ℕ) : n = 867 → n / 37 = 23 ∧ n % 37 = 16 := by
  sorry

end division_problem_l1749_174991


namespace sms_authenticity_l1749_174972

/-- Represents an SMS message -/
structure SMS where
  content : String
  sender : String

/-- Represents a bank card -/
structure BankCard where
  number : String
  bank : String
  officialPhoneNumber : String

/-- Represents a bank's SMS characteristics -/
structure BankSMSCharacteristics where
  shortNumber : String
  messageFormat : String → Bool

/-- Determines if an SMS is genuine based on comparison and bank confirmation -/
def isGenuineSMS (message : SMS) (card : BankCard) (prevMessages : List SMS) 
                 (bankCharacteristics : BankSMSCharacteristics) : Prop :=
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true)

/-- Main theorem: An SMS is genuine iff it matches previous messages and is confirmed by the bank -/
theorem sms_authenticity 
  (message : SMS) 
  (card : BankCard) 
  (prevMessages : List SMS) 
  (bankCharacteristics : BankSMSCharacteristics) :
  isGenuineSMS message card prevMessages bankCharacteristics ↔ 
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true) :=
by sorry


end sms_authenticity_l1749_174972


namespace recorder_price_problem_l1749_174939

theorem recorder_price_problem :
  ∀ (a b : ℕ),
    a < 5 ∧ b < 10 →  -- Ensure the old price is less than 50
    (10 * b + a : ℚ) = (10 * a + b : ℚ) * (6/5) →  -- 20% increase and digit swap
    10 * b + a = 54 :=
by sorry

end recorder_price_problem_l1749_174939


namespace C_excircle_touches_circumcircle_l1749_174927

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the semiperimeter of a triangle
def semiperimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the C-excircle of a triangle
def C_excircle (t : Triangle) : Circle := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Circle := sorry

-- Define tangency between two circles
def are_tangent (c1 c2 : Circle) : Prop := sorry

-- Theorem statement
theorem C_excircle_touches_circumcircle 
  (ABC : Triangle) 
  (p : ℝ) 
  (E F : Point) :
  semiperimeter ABC = p →
  E.x ≤ F.x →
  distance ABC.A E + distance E F + distance F ABC.B = distance ABC.A ABC.B →
  distance ABC.C E = p →
  distance ABC.C F = p →
  are_tangent (C_excircle ABC) (circumcircle (Triangle.mk E F ABC.C)) :=
sorry

end C_excircle_touches_circumcircle_l1749_174927


namespace x_squared_plus_y_squared_equals_four_l1749_174977

theorem x_squared_plus_y_squared_equals_four
  (x y : ℝ)
  (h1 : x^3 = 3*y^2*x + 5 - Real.sqrt 7)
  (h2 : y^3 = 3*x^2*y + 5 + Real.sqrt 7) :
  x^2 + y^2 = 4 := by sorry

end x_squared_plus_y_squared_equals_four_l1749_174977


namespace marissa_boxes_tied_l1749_174934

/-- The number of boxes tied with a given amount of ribbon -/
def boxes_tied (total_ribbon leftover_ribbon ribbon_per_box : ℚ) : ℚ :=
  (total_ribbon - leftover_ribbon) / ribbon_per_box

/-- Theorem: Given the conditions, Marissa tied 5 boxes -/
theorem marissa_boxes_tied :
  boxes_tied 4.5 1 0.7 = 5 := by
  sorry

end marissa_boxes_tied_l1749_174934


namespace print_shop_charge_l1749_174938

/-- The charge per color copy at print shop X -/
def charge_x : ℝ := 1.25

/-- The charge per color copy at print shop Y -/
def charge_y : ℝ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 80

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 120

theorem print_shop_charge : 
  charge_x * num_copies + additional_charge = charge_y * num_copies := by
  sorry

#check print_shop_charge

end print_shop_charge_l1749_174938


namespace log_sum_simplification_l1749_174909

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 18 + 1) +
  1 / (Real.log 2 / Real.log 12 + 1) +
  1 / (Real.log 7 / Real.log 8 + 1) =
  13 / 12 := by sorry

end log_sum_simplification_l1749_174909


namespace hyperbola_eccentricity_l1749_174943

/-- Given a hyperbola with the following properties:
    - The distance from the vertex to its asymptote is 2
    - The distance from the focus to the asymptote is 6
    Then the eccentricity of the hyperbola is 3 -/
theorem hyperbola_eccentricity (vertex_to_asymptote : ℝ) (focus_to_asymptote : ℝ) 
  (h1 : vertex_to_asymptote = 2)
  (h2 : focus_to_asymptote = 6) :
  let e := focus_to_asymptote / vertex_to_asymptote
  e = 3 := by sorry

end hyperbola_eccentricity_l1749_174943


namespace flagpole_break_height_l1749_174926

/-- Given a flagpole of height 8 meters that breaks and touches the ground 3 meters from its base,
    the height of the break point is √3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 8 →  -- Original height of flagpole
  x^2 + 3^2 = (h - x)^2 →  -- Pythagorean theorem application
  x = Real.sqrt 3 :=
by sorry

end flagpole_break_height_l1749_174926


namespace square_field_side_length_l1749_174929

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 256 → side ^ 2 = area → side = 16 := by sorry

end square_field_side_length_l1749_174929


namespace five_digit_numbers_count_l1749_174911

/-- The number of odd digits available -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available -/
def num_even_digits : ℕ := 5

/-- The total number of digits in the formed numbers -/
def total_digits : ℕ := 5

/-- Function to calculate the number of ways to form five-digit numbers -/
def count_five_digit_numbers : ℕ :=
  let case1 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 3 * Nat.factorial total_digits
  let case2 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 2 * Nat.choose 4 1 * Nat.factorial 4
  case1 + case2

/-- Theorem stating that the number of unique five-digit numbers is 10,560 -/
theorem five_digit_numbers_count : count_five_digit_numbers = 10560 := by
  sorry

end five_digit_numbers_count_l1749_174911


namespace smallest_angle_tan_equation_l1749_174994

open Real

theorem smallest_angle_tan_equation (x : ℝ) : 
  (0 < x) ∧ 
  (x < 2 * π) ∧
  (tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) →
  x = 5.625 * (π / 180) :=
by sorry

end smallest_angle_tan_equation_l1749_174994


namespace parabola_hyperbola_coincidence_l1749_174976

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with
    the right vertex of the hyperbola x^2/4 - y^2 = 1 -/
theorem parabola_hyperbola_coincidence (p : ℝ) : 
  (∃ x y : ℝ, y^2 = 2*p*x ∧ x^2/4 - y^2 = 1 ∧ x = p/2 ∧ y = 0) → p = 4 :=
by sorry

end parabola_hyperbola_coincidence_l1749_174976


namespace uncle_wang_flower_pots_l1749_174903

theorem uncle_wang_flower_pots :
  ∃! x : ℕ,
    ∃ a : ℕ,
      x / 2 + x / 4 + x / 7 + a = x ∧
      1 ≤ a ∧ a < 6 :=
by
  sorry

end uncle_wang_flower_pots_l1749_174903


namespace inequality_proof_l1749_174950

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end inequality_proof_l1749_174950


namespace penny_species_count_l1749_174997

/-- The number of shark species Penny identified -/
def shark_species : ℕ := 35

/-- The number of eel species Penny identified -/
def eel_species : ℕ := 15

/-- The number of whale species Penny identified -/
def whale_species : ℕ := 5

/-- The total number of species Penny identified -/
def total_species : ℕ := shark_species + eel_species + whale_species

/-- Theorem stating that the total number of species Penny identified is 55 -/
theorem penny_species_count : total_species = 55 := by
  sorry

end penny_species_count_l1749_174997


namespace tan_alpha_equals_three_l1749_174970

theorem tan_alpha_equals_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10) : 
  Real.tan α = 3 := by
sorry

end tan_alpha_equals_three_l1749_174970


namespace solution_volume_proof_l1749_174979

-- Define the volume of pure acid in liters
def pure_acid_volume : ℝ := 1.6

-- Define the concentration of the solution as a percentage
def solution_concentration : ℝ := 20

-- Define the total volume of the solution in liters
def total_solution_volume : ℝ := 8

-- Theorem to prove
theorem solution_volume_proof :
  pure_acid_volume = (solution_concentration / 100) * total_solution_volume :=
by sorry

end solution_volume_proof_l1749_174979


namespace sqrt_two_cos_sin_equality_l1749_174913

theorem sqrt_two_cos_sin_equality (x : ℝ) :
  Real.sqrt 2 * (Real.cos (2 * x))^4 - Real.sqrt 2 * (Real.sin (2 * x))^4 = Real.cos (2 * x) + Real.sin (2 * x) →
  ∃ k : ℤ, x = Real.pi * (4 * k - 1) / 8 := by
sorry

end sqrt_two_cos_sin_equality_l1749_174913


namespace complex_equation_solution_l1749_174982

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l1749_174982


namespace second_caterer_cheaper_at_31_l1749_174930

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basic_fee : ℕ
  per_person : ℕ
  additional_fee : ℕ

/-- The first caterer's pricing model -/
def caterer1 : CatererPricing := {
  basic_fee := 150,
  per_person := 20,
  additional_fee := 0
}

/-- The second caterer's pricing model -/
def caterer2 : CatererPricing := {
  basic_fee := 250,
  per_person := 15,
  additional_fee := 50
}

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people + c.additional_fee

/-- Theorem stating that 31 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_31 :
  (∀ n : ℕ, n < 31 → total_cost caterer1 n ≤ total_cost caterer2 n) ∧
  (total_cost caterer1 31 > total_cost caterer2 31) :=
sorry

end second_caterer_cheaper_at_31_l1749_174930


namespace odd_function_negative_domain_l1749_174992

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end odd_function_negative_domain_l1749_174992


namespace relationship_abxy_l1749_174912

theorem relationship_abxy (a b x y : ℚ) 
  (eq1 : x + y = a + b) 
  (ineq1 : y - x < a - b) 
  (ineq2 : b > a) : 
  y < a ∧ a < b ∧ b < x :=
sorry

end relationship_abxy_l1749_174912


namespace function_properties_l1749_174923

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem function_properties (a b : ℝ) :
  (∃ y, f a b 1 = y ∧ x + y - 3 = 0) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 8) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≥ -4) ∧
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, ∃ y ∈ Set.Ioo (-1 : ℝ) 1, x < y ∧ f a b x > f a b y) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
by sorry

end function_properties_l1749_174923


namespace boat_speed_l1749_174962

theorem boat_speed (current_speed : ℝ) (downstream_distance : ℝ) (time : ℝ) :
  current_speed = 3 →
  downstream_distance = 6.75 →
  time = 0.25 →
  ∃ (boat_speed : ℝ),
    boat_speed = 24 ∧
    downstream_distance = (boat_speed + current_speed) * time :=
by
  sorry

end boat_speed_l1749_174962


namespace fraction_inequality_counterexample_l1749_174995

theorem fraction_inequality_counterexample :
  ∃ (a b c d A B C D : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
    a/b > A/B ∧ 
    c/d > C/D ∧ 
    (a+c)/(b+d) ≤ (A+C)/(B+D) := by
  sorry

end fraction_inequality_counterexample_l1749_174995


namespace geometric_sequence_property_l1749_174925

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 2 * a 4 = 4) :
  a 1 * a 5 + a 3 = 6 := by
  sorry


end geometric_sequence_property_l1749_174925


namespace larger_number_proof_l1749_174922

theorem larger_number_proof (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := by
  sorry

end larger_number_proof_l1749_174922


namespace bleacher_sets_l1749_174969

theorem bleacher_sets (total_fans : ℕ) (fans_per_set : ℕ) (h1 : total_fans = 2436) (h2 : fans_per_set = 812) :
  total_fans / fans_per_set = 3 :=
by sorry

end bleacher_sets_l1749_174969


namespace pebbles_on_day_15_l1749_174989

/-- Represents Murtha's pebble collection strategy -/
def pebbleCollection (n : ℕ) : ℕ := 
  if n < 15 then n else 2 * 15

/-- The sum of pebbles collected up to day n -/
def totalPebbles (n : ℕ) : ℕ := 
  (List.range n).map pebbleCollection |>.sum

/-- Theorem stating the total number of pebbles collected by the end of the 15th day -/
theorem pebbles_on_day_15 : totalPebbles 15 = 135 := by
  sorry

end pebbles_on_day_15_l1749_174989


namespace unequal_gender_probability_l1749_174961

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male (or female) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of males and females
    given n independently determined genders with probability p of being male -/
def unequal_probability (n : ℕ) (p : ℚ) : ℚ :=
  1 - (n.choose (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)

/-- The theorem to be proved -/
theorem unequal_gender_probability :
  unequal_probability n p = 793/1024 := by
  sorry

end unequal_gender_probability_l1749_174961


namespace triangle_3_4_5_l1749_174954

/-- A triangle can be formed from three line segments if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the line segments 3, 4, and 5 can form a triangle. -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end triangle_3_4_5_l1749_174954


namespace max_value_of_exponential_difference_l1749_174999

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end max_value_of_exponential_difference_l1749_174999


namespace rectangular_prism_sum_l1749_174984

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  rp.faces + rp.edges + rp.vertices = 26 :=
by sorry

end rectangular_prism_sum_l1749_174984


namespace banana_orange_equivalence_l1749_174967

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 9 bananas are worth 3 oranges -/
theorem banana_orange_equivalence 
  (h : (3/4 : ℚ) * 12 * (banana_value : ℚ) = 9 * (orange_value : ℚ)) :
  (1/3 : ℚ) * 9 * banana_value = 3 * orange_value :=
by sorry


end banana_orange_equivalence_l1749_174967


namespace min_sum_squares_l1749_174915

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 7200 / 13 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 7200 / 13 :=
sorry

end min_sum_squares_l1749_174915


namespace unique_non_range_value_l1749_174975

/-- The function g defined as (px + q) / (rx + s) -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- The theorem stating that 30 is the unique number not in the range of g -/
theorem unique_non_range_value
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_11 : g p q r s 11 = 11)
  (h_41 : g p q r s 41 = 41)
  (h_inverse : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, ∀ x, g p q r s x ≠ y ∧ y = 30 :=
sorry

end unique_non_range_value_l1749_174975


namespace pure_imaginary_fraction_l1749_174959

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 →
  a = 1 := by
sorry

end pure_imaginary_fraction_l1749_174959


namespace max_x_minus_y_l1749_174968

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (m : ℝ), m = 1 / (2 * Real.sqrt 3) ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → (a - b) ≤ m :=
by sorry

end max_x_minus_y_l1749_174968


namespace brick_wall_theorem_l1749_174902

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def totalBricks (rows : ℕ) (bottomRowBricks : ℕ) : ℕ :=
  (rows * (2 * bottomRowBricks - rows + 1)) / 2

/-- Theorem: A brick wall with 5 rows, where the bottom row has 38 bricks
    and each subsequent row has one less brick than the row below it,
    contains a total of 180 bricks. -/
theorem brick_wall_theorem :
  totalBricks 5 38 = 180 := by
  sorry

end brick_wall_theorem_l1749_174902


namespace simplify_expression_l1749_174945

theorem simplify_expression (x y : ℝ) : (3*x)^4 + (4*x)*(x^3) + (5*y)^2 = 85*x^4 + 25*y^2 := by
  sorry

end simplify_expression_l1749_174945


namespace complex_expression_equals_nine_l1749_174965

theorem complex_expression_equals_nine :
  (0.4 + 8 * (5 - 0.8 * (5 / 8)) - 5 / (2 + 1 / 2)) /
  ((1 + 7 / 8) * 8 - (8.9 - 2.6 / (2 / 3))) * (34 + 2 / 5) * 90 = 9 := by
  sorry

end complex_expression_equals_nine_l1749_174965


namespace complex_multiplication_l1749_174941

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number 3+2i -/
def z : ℂ := 3 + 2 * i

theorem complex_multiplication :
  z * i = -2 + 3 * i := by sorry

end complex_multiplication_l1749_174941


namespace simplify_complex_expression_l1749_174901

theorem simplify_complex_expression (x : ℝ) (hx : x > 0) : 
  Real.sqrt (2 * (1 + Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2))) = (x^2 + 1) / x := by
  sorry

end simplify_complex_expression_l1749_174901


namespace least_value_quadratic_inequality_l1749_174933

theorem least_value_quadratic_inequality :
  (∀ b : ℝ, b < 4 → -b^2 + 9*b - 20 < 0) ∧
  (-4^2 + 9*4 - 20 = 0) ∧
  (∀ b : ℝ, b > 4 → -b^2 + 9*b - 20 ≤ -b^2 + 9*4 - 20) :=
by sorry

end least_value_quadratic_inequality_l1749_174933


namespace cube_volume_division_l1749_174966

theorem cube_volume_division (V : ℝ) (a b : ℝ) (h1 : V > 0) (h2 : b > a) (h3 : a > 0) :
  let diagonal_ratio := a / (b - a)
  let volume_ratio := a^3 / (b^3 - a^3)
  ∃ (V1 V2 : ℝ), V1 + V2 = V ∧ V1 / V2 = volume_ratio :=
sorry

end cube_volume_division_l1749_174966
