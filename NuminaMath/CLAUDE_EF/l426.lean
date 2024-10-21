import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_1_5_km_l426_42653

/-- The length of a bridge in kilometers. -/
noncomputable def bridge_length (walking_speed : ℝ) (crossing_time_minutes : ℝ) : ℝ :=
  walking_speed * (crossing_time_minutes / 60)

/-- Theorem stating that a bridge's length is 1.5 km when crossed by a man walking at 6 km/hr in 15 minutes. -/
theorem bridge_length_is_1_5_km :
  bridge_length 6 15 = 1.5 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that 6 * (15 / 60) = 1.5
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_1_5_km_l426_42653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_230_l426_42675

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Theorem stating that the length of the bridge is 230 meters under the given conditions. -/
theorem bridge_length_is_230 :
  bridge_length 145 45 30 = 230 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval bridge_length 145 45 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_230_l426_42675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_calculation_l426_42639

/-- Calculates the stock quote given investment details -/
noncomputable def calculate_stock_quote (investment : ℝ) (dividend_rate : ℝ) (income : ℝ) : ℝ :=
  let face_value := (income * 100) / dividend_rate
  (investment / face_value) * 100

/-- Theorem stating the stock quote calculation -/
theorem stock_quote_calculation (investment : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 1800)
  (h2 : dividend_rate = 9)
  (h3 : income = 120) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (calculate_stock_quote investment dividend_rate income - 135) < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_calculation_l426_42639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_power_of_three_l426_42611

/-- Defines the sequence with the given properties -/
def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 7 ∧ ∀ n ≥ 2, a n ^ 2 + 5 = a (n - 1) * a (n + 1)

/-- Theorem stating the main result -/
theorem sequence_prime_power_of_three (a : ℕ → ℤ) (n : ℕ) :
  my_sequence a →
  Nat.Prime (Int.natAbs (a n + (-1)^n)) →
  ∃ m : ℕ, n = 3^m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_power_of_three_l426_42611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_under_translation_l426_42658

noncomputable def sample_a : List ℝ := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
noncomputable def sample_b : List ℝ := sample_a.map (· + 2)

noncomputable def standard_deviation (sample : List ℝ) : ℝ :=
  let mean := sample.sum / sample.length
  Real.sqrt ((sample.map (fun x => (x - mean) ^ 2)).sum / sample.length)

theorem standard_deviation_invariant_under_translation :
  standard_deviation sample_a = standard_deviation sample_b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_under_translation_l426_42658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l426_42659

theorem vector_problems (θ : Real) :
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (2, -1)
  -- Part 1
  (a.1 * b.1 + a.2 * b.2 = 0 →
    (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1/3) ∧
  -- Part 2
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 ∧ 0 < θ ∧ θ < Real.pi/2 →
    Real.sin (θ + Real.pi/4) = 7 * Real.sqrt 2 / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l426_42659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l426_42644

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the problem statement
theorem problem_statement (n : ℝ) : 
  floor 6.5 * floor (2/3 : ℝ) + floor 2 * (7.2 : ℝ) + floor n - (6.6 : ℝ) = (15.8 : ℝ) → 
  floor n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l426_42644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_L_l426_42638

/-- The curve C in the Cartesian coordinate system -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line l in the Cartesian coordinate system -/
def L (x y : ℝ) : Prop := x - 2*y - 6 = 0

/-- The distance from a point (x, y) to the line L -/
noncomputable def distance_to_L (x y : ℝ) : ℝ := 
  |x - 2*y - 6| / Real.sqrt 5

/-- The maximum distance from any point on curve C to line L is 2√5 -/
theorem max_distance_C_to_L : 
  ∀ x y : ℝ, C x y → ∃ x' y' : ℝ, C x' y' ∧ 
    distance_to_L x' y' = 2 * Real.sqrt 5 ∧
    ∀ x'' y'' : ℝ, C x'' y'' → distance_to_L x'' y'' ≤ 2 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_L_l426_42638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_of_given_circles_l426_42679

/-- The common chord length of two circles -/
def common_chord_length (c1 c2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- Theorem stating the common chord length of the given circles -/
theorem common_chord_length_of_given_circles :
  common_chord_length circle1 circle2 = Real.sqrt 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_of_given_circles_l426_42679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_second_in_physics_l426_42696

-- Define the students
inductive Student : Type where
  | Kolya : Student
  | Zhenya : Student
  | Nadya : Student
deriving Repr, DecidableEq

-- Define the exams
structure Exam where
  name : String
  first : Student
  second : Student
  third : Student

-- Define the point system
def points (place : Nat) : Nat :=
  match place with
  | 1 => 5
  | 2 => 2
  | 3 => 1
  | _ => 0

-- Define the total points for each student
def total_points (s : Student) (exams : List Exam) : Nat :=
  exams.foldl (fun acc exam =>
    acc + if exam.first = s then points 1
          else if exam.second = s then points 2
          else if exam.third = s then points 3
          else 0) 0

-- Theorem statement
theorem kolya_second_in_physics 
  (exams : List Exam) 
  (h1 : exams.length = 5)
  (h2 : ∃ exam ∈ exams, exam.name = "Algebra" ∧ exam.first = Student.Zhenya)
  (h3 : total_points Student.Kolya exams = 22)
  (h4 : total_points Student.Zhenya exams = 9)
  (h5 : total_points Student.Nadya exams = 9)
  : ∃ exam ∈ exams, exam.name = "Physics" ∧ exam.second = Student.Kolya := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_second_in_physics_l426_42696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_l426_42645

def R : ℝ × ℝ := (10, 8)

def line1 (x y : ℝ) : Prop := 9 * y = 18 * x

def line2 (x y : ℝ) : Prop := 15 * y = 6 * x

def is_midpoint (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

theorem PQ_length (P Q : ℝ × ℝ) :
  line1 P.1 P.2 →
  line2 Q.1 Q.2 →
  is_midpoint P Q R →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 116 :=
by sorry

#check PQ_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_l426_42645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l426_42687

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Calculates the time it takes for two trains to completely pass each other -/
noncomputable def passingTime (trainA trainB : Train) : ℝ :=
  (trainA.length + trainB.length) / (trainA.speed + trainB.speed)

/-- Theorem stating the time it takes for the given trains to pass each other -/
theorem trains_passing_time :
  let trainA : Train := { length := 80, speed := 36 * (5/18) }
  let trainB : Train := { length := 120, speed := 45 * (5/18) }
  abs (passingTime trainA trainB - 8.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l426_42687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l426_42637

/-- The y-coordinate of the third vertex of an equilateral triangle -/
noncomputable def third_vertex_y_coord (x1 y1 x2 y2 : ℝ) : ℝ :=
  let side_length := |x2 - x1|
  y1 + side_length * (Real.sqrt 3 / 2)

/-- Theorem stating the y-coordinate of the third vertex of the equilateral triangle -/
theorem equilateral_triangle_third_vertex
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 0 ∧ y1 = 7)  -- First vertex at (0,7)
  (h2 : x2 = 10 ∧ y2 = 7)  -- Second vertex at (10,7)
  (h3 : x2 > x1)  -- Ensures the side length calculation is correct
  : third_vertex_y_coord x1 y1 x2 y2 = 7 + 5 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l426_42637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_interpretation_plaza_area_interpretation_l426_42636

/-- Represents the donation of a single student in dollars -/
def m : ℝ := sorry

/-- Represents the donation of another student in dollars -/
def n : ℝ := sorry

/-- Represents the side length of a square plaza in meters -/
def a : ℝ := sorry

/-- The total donation from 5 students donating m dollars each and 2 students donating n dollars each -/
def total_donation : ℝ := 5 * m + 2 * n

/-- The area of a single square plaza with side length a -/
def plaza_area : ℝ := a ^ 2

theorem donation_interpretation :
  total_donation = 5 * m + 2 * n :=
by simp [total_donation]

theorem plaza_area_interpretation :
  6 * plaza_area = 6 * (a ^ 2) :=
by simp [plaza_area]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_interpretation_plaza_area_interpretation_l426_42636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blanket_average_price_l426_42670

theorem blanket_average_price : 
  let blanket_group1_count : ℕ := 4
  let blanket_group1_price : ℚ := 100
  let blanket_group2_count : ℕ := 5
  let blanket_group2_price : ℚ := 150
  let blanket_group3_count : ℕ := 2
  let blanket_group3_price : ℚ := 350
  let total_blankets : ℕ := blanket_group1_count + blanket_group2_count + blanket_group3_count
  let total_cost : ℚ := blanket_group1_count * blanket_group1_price + 
                        blanket_group2_count * blanket_group2_price + 
                        blanket_group3_count * blanket_group3_price
  let average_price : ℚ := total_cost / total_blankets
  ∃ (rounded_average : ℚ), (abs (rounded_average - 168.18) < 0.01) ∧ (abs (rounded_average - average_price) < 0.01)
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blanket_average_price_l426_42670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l426_42691

/-- Represents the probability of rolling an even number on the unfair die -/
noncomputable def p_even : ℝ := 1 / 3

/-- Represents the probability of rolling an odd number on the unfair die -/
noncomputable def p_odd : ℝ := 2 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 3

/-- Theorem stating that the probability of getting an even sum when rolling the unfair die three times is 1 -/
theorem even_sum_probability : 
  (p_even ^ 3) + (p_odd ^ 3) + 
  (3 * p_even * p_odd ^ 2) + 
  (3 * p_even ^ 2 * p_odd) = 1 := by
  sorry

#check even_sum_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l426_42691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_squares_characterization_l426_42635

/-- A number is a repeated-digit square if it's a square number and its decimal representation consists of the same digit repeated. -/
def is_repeated_digit_square (n : ℕ) : Prop :=
  ∃ (d : ℕ) (k : ℕ), 
    n = d * (10^k - 1) / 9 ∧ 
    d < 10 ∧ 
    ∃ (m : ℕ), n = m^2

/-- The set of all repeated-digit square numbers -/
def repeated_digit_squares : Set ℕ :=
  {n : ℕ | is_repeated_digit_square n}

/-- Theorem stating that the only repeated-digit square numbers are 0, 1, 4, and 9 -/
theorem repeated_digit_squares_characterization : 
  repeated_digit_squares = {0, 1, 4, 9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_squares_characterization_l426_42635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_monotone_intervals_l426_42634

/-- The function f(x) = sin x + cos x is monotonically increasing on the intervals [2kπ - 3π/4, 2kπ + π/4] for all integers k. -/
theorem sin_plus_cos_monotone_intervals (k : ℤ) :
  StrictMonoOn (λ x => Real.sin x + Real.cos x) (Set.Icc (2 * k * Real.pi - 3 * Real.pi / 4) (2 * k * Real.pi + Real.pi / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_monotone_intervals_l426_42634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_data_properties_l426_42621

/-- Data point representing time and sales volume -/
structure DataPoint where
  x : ℝ  -- Time in months
  y : ℝ  -- Sales volume in ten thousand units

/-- Linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

def data : List DataPoint := [
  ⟨1, 1⟩, ⟨2, 1.6⟩, ⟨3, 2.0⟩, ⟨4, 2.4⟩, ⟨5, 3⟩
]

def regression : LinearRegression := ⟨0.48, 0.56⟩

/-- Calculate the sample centroid of a list of data points -/
noncomputable def sampleCentroid (data : List DataPoint) : DataPoint :=
  let n := data.length
  let sumX := data.foldl (fun sum p => sum + p.x) 0
  let sumY := data.foldl (fun sum p => sum + p.y) 0
  ⟨sumX / n, sumY / n⟩

/-- Check if two real numbers are approximately equal -/
noncomputable def approxEqual (a b : ℝ) (ε : ℝ := 1e-6) : Bool :=
  abs (a - b) < ε

theorem sales_data_properties : 
  let centroid := sampleCentroid data
  (data[3].y = 2.4) ∧ 
  (approxEqual centroid.x 3 ∧ approxEqual centroid.y 2.0) ∧
  (regression.slope > 0) := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_data_properties_l426_42621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l426_42683

open Real

theorem triangle_property (A B C a b c : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sum : A + B + C = π)
  (h_tan : Real.sqrt 3 * (Real.tan A - Real.tan B) = 1 + Real.tan A * Real.tan B)
  (h_sides : a^2 - a*b = c^2 - b^2) :
  A = 5*π/12 ∧ B = π/4 ∧ C = π/3 ∧
  1 ≤ ‖(3 * Real.sin A, 3 * Real.cos A) - (2 * Real.cos B, 2 * Real.sin B)‖ ∧
  ‖(3 * Real.sin A, 3 * Real.cos A) - (2 * Real.cos B, 2 * Real.sin B)‖ < Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l426_42683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_interval_l426_42648

/-- The general term of the power series -/
noncomputable def a (n : ℕ) (x : ℝ) : ℝ := (-1)^n * x^n / ((n + 1 : ℝ) * 2^n)

/-- The power series -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, a n x

/-- The interval of convergence -/
def I : Set ℝ := Set.Ioo (-2) 2

theorem convergence_interval :
  (∀ x ∈ I, HasSum (fun n ↦ a n x) (S x)) ∧
  (∀ x ∉ I, ¬ Summable (fun n ↦ a n x)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_interval_l426_42648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_alphabets_written_l426_42692

/-- The set of vowels in the English alphabet -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- The number of times each vowel is written -/
def TimesWritten : ℕ := 4

/-- Theorem: The total number of alphabets written on the board is 20 -/
theorem total_alphabets_written : 
  Vowels.card * TimesWritten = 20 := by
  -- Calculate the cardinality of the Vowels set
  have h1 : Vowels.card = 5 := by rfl
  -- Multiply by TimesWritten
  have h2 : 5 * TimesWritten = 20 := by rfl
  -- Combine the steps
  rw [h1, h2]

#eval Vowels.card * TimesWritten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_alphabets_written_l426_42692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_correct_l426_42618

def chess_club_mixed_groups 
  (total_children : ℕ) 
  (total_groups : ℕ) 
  (group_size : ℕ) 
  (boy_games : ℕ) 
  (girl_games : ℕ) : ℕ :=
  let total_games := total_groups * (group_size * (group_size - 1) / 2)
  let mixed_games := total_games - boy_games - girl_games
  mixed_games / 2

#eval chess_club_mixed_groups 90 30 3 30 14

theorem chess_club_mixed_groups_correct : 
  chess_club_mixed_groups 90 30 3 30 14 = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_correct_l426_42618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_obliques_method_area_l426_42604

/-- The area of a triangle calculated using the "three obliques method" -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2))

/-- Theorem stating that for a triangle with sides 3, 7, and 8, 
    the area calculated using the "three obliques method" is 6√3 -/
theorem three_obliques_method_area : 
  triangleArea 3 7 8 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_obliques_method_area_l426_42604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_256_is_saturday_l426_42698

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day of the week that is n days after the given day -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

theorem day_256_is_saturday (h : dayAfter DayOfWeek.Sunday 40 = DayOfWeek.Sunday) :
  dayAfter DayOfWeek.Sunday 256 = DayOfWeek.Saturday := by
  sorry

#eval dayAfter DayOfWeek.Sunday 256

end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_256_is_saturday_l426_42698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_not_in_domain_l426_42688

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1/x))

-- Define the set of x values not in the domain of g
def not_in_domain (x : ℝ) : Prop :=
  x = 0 ∨ x = -1/2 ∨ x = -2/5

-- Theorem statement
theorem sum_of_x_not_in_domain :
  Finset.sum {0, -1/2, -2/5} id = -19/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_not_in_domain_l426_42688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_of_derivatives_l426_42669

open Real

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a*x + m

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

-- Theorem statement
theorem negative_product_of_derivatives (a m : ℝ) (h_a : 0 < a) (h_a1 : a < 1) :
  ∃ t : ℝ, f' a t < 0 ∧ f' a (t + 2) * f' a ((2*t + 1) / 3) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_of_derivatives_l426_42669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l426_42695

noncomputable def f (x : ℝ) := Real.log (3 - 4 * Real.sin x ^ 2)

def domain (k : ℤ) : Set ℝ :=
  Set.Ioo (2 * k * Real.pi - Real.pi / 3) (2 * k * Real.pi + Real.pi / 3) ∪
  Set.Ioo (2 * k * Real.pi + 2 * Real.pi / 3) (2 * k * Real.pi + 4 * Real.pi / 3)

theorem f_domain :
  {x : ℝ | ∃ y, f x = y} = ⋃ k, domain k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l426_42695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_new_cakes_l426_42668

/-- Given that Baker initially had 173 cakes, sold 86 cakes, and now has 190 cakes,
    prove that he bought 103 new cakes. -/
theorem baker_new_cakes : ℕ := by
  let initial_cakes : ℕ := 173
  let sold_cakes : ℕ := 86
  let final_cakes : ℕ := 190
  let new_cakes : ℕ := final_cakes + sold_cakes - initial_cakes
  have h : new_cakes = 103 := by
    -- Proof goes here
    sorry
  exact new_cakes


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_new_cakes_l426_42668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l426_42662

theorem simplify_expression (a : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) * (6 * a^5) * (Nat.factorial 7) = 3628800 * a^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l426_42662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l426_42694

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- The vertex of a parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℚ × ℚ :=
  let x := -p.b / (2 * p.a)
  (x, p.y_at x)

theorem parabola_c_value (p : Parabola) :
  p.vertex = (2, 3) → p.y_at 0 = 4 → p.c = 4 := by
  sorry

#check parabola_c_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l426_42694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_divisor_l426_42666

-- Define the polynomial P(x)
def P (x : ℕ) : ℕ := sorry

-- Define the sequence b_n
def b : ℕ → ℕ
  | 0 => P 0  -- b₀ = a₀ (coefficient of x⁰ in P(x))
  | n + 1 => P (b n)

-- State the theorem
theorem exists_prime_divisor (d : ℕ) (hd : d ≥ 2) :
  ∀ n ≥ 2, ∃ p : ℕ, Nat.Prime p ∧ p ∣ b n ∧ ¬(p ∣ (Finset.prod (Finset.range (n-1)) (λ i => b (i+1)))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_divisor_l426_42666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_inequality_proof_l426_42660

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a * x) * Real.log (1 + x) - x

-- Theorem 1
theorem min_value_f (a : ℝ) (h : a ≤ -1/2) :
  ∀ x ∈ Set.Icc 0 1, f a x ≥ 0 := by sorry

-- Theorem 2
theorem inequality_proof :
  (2019/2018: ℝ)^(2018 * (1/2)) > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_inequality_proof_l426_42660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_rod_volume_l426_42652

/-- The volume of a cylindrical rod with given conditions -/
theorem cylindrical_rod_volume
  (length : ℝ)
  (surface_area_increase : ℝ)
  (h1 : length = 2)
  (h2 : surface_area_increase = 0.6) :
  let radius := Real.sqrt (surface_area_increase / (4 * Real.pi))
  let volume := Real.pi * radius^2 * length
  volume = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_rod_volume_l426_42652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_ratio_l426_42609

def z₁ : ℂ := Complex.mk 1 (-2)

theorem imaginary_part_of_ratio (z₂ : ℂ) 
  (h : z₂.re = -z₁.re ∧ z₂.im = z₁.im) : 
  Complex.im (z₂ / z₁) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_ratio_l426_42609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_red_tint_percentage_is_33_33_percent_l426_42613

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  total_volume : ℝ
  red_tint_percent : ℝ
  yellow_tint_percent : ℝ
  water_percent : ℝ

/-- Calculates the new red tint percentage after adding more red tint -/
noncomputable def new_red_tint_percentage (original : PaintMixture) (added_red_tint : ℝ) : ℝ :=
  let original_red_tint := original.total_volume * (original.red_tint_percent / 100)
  let new_total_volume := original.total_volume + added_red_tint
  let new_red_tint := original_red_tint + added_red_tint
  (new_red_tint / new_total_volume) * 100

/-- Theorem: The new red tint percentage is approximately 33.33% -/
theorem new_red_tint_percentage_is_33_33_percent 
  (original : PaintMixture)
  (h1 : original.total_volume = 40)
  (h2 : original.red_tint_percent = 20)
  (h3 : original.yellow_tint_percent = 40)
  (h4 : original.water_percent = 40)
  (h5 : original.red_tint_percent + original.yellow_tint_percent + original.water_percent = 100)
  (added_red_tint : ℝ)
  (h6 : added_red_tint = 8) :
  abs (new_red_tint_percentage original added_red_tint - 33.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_red_tint_percentage_is_33_33_percent_l426_42613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dolls_count_l426_42629

theorem dolls_count (total_toys : ℕ) (action_figure_ratio : ℚ) (dolls : ℕ) : 
  total_toys = 24 →
  action_figure_ratio = 1/4 →
  dolls = total_toys - (action_figure_ratio * ↑total_toys).floor →
  dolls = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dolls_count_l426_42629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l426_42655

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (sin x - cos x) / Real.log (1/3)

-- Define the monotonic increasing interval
def monotonic_increasing_interval (k : ℤ) : Set ℝ := 
  Set.Ioo (2 * k * π + 3 * π / 4) (2 * k * π + 5 * π / 4)

-- State the theorem
theorem f_monotonic_increasing_interval (k : ℤ) :
  StrictMonoOn f (monotonic_increasing_interval k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l426_42655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l426_42654

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

-- Theorem statement
theorem f_properties :
  -- 1. The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- 2. The minimum value on [0, π/2] is 0
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 0 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ f x) ∧
  -- 3. The maximum value on [0, π/2] is 1 + √2
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1 + Real.sqrt 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l426_42654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_value_of_a_range_of_b_l426_42651

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-x) + a * x - 1 / x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (-x) + 2 * x

-- Theorem for part I
theorem extremum_value_of_a :
  ∃ (a : ℝ), ∀ (x : ℝ), x ≠ 0 → deriv (f a) x = 0 ↔ x = -1 :=
sorry

-- Theorem for part II
theorem range_of_b (a : ℝ) (b : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g a x₁ = b ∧ g a x₂ = b) →
  b > 3 - Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_value_of_a_range_of_b_l426_42651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_perimeter_of_cut_pieces_l426_42615

/-- Represents the triangle and its properties --/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Represents a piece of the cut triangle --/
structure TrianglePiece where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle piece --/
noncomputable def area (piece : TrianglePiece) : ℝ :=
  1/2 * piece.base * piece.height

/-- Calculates the perimeter of a triangle piece --/
noncomputable def perimeter (piece : TrianglePiece) : ℝ :=
  piece.base + 2 * Real.sqrt (piece.height^2 + (piece.base/2)^2)

/-- Theorem stating the greatest perimeter among the cut pieces --/
theorem greatest_perimeter_of_cut_pieces (triangle : IsoscelesTriangle)
  (h_base : triangle.base = 10)
  (h_height : triangle.height = 12)
  : ∃ (piece : TrianglePiece),
    area piece = (1/2 * triangle.base * triangle.height) / 10 ∧
    ∀ (other : TrianglePiece),
      area other = (1/2 * triangle.base * triangle.height) / 10 →
      perimeter other ≤ 5 + 2 * Real.sqrt (144/25 + 25/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_perimeter_of_cut_pieces_l426_42615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l426_42608

noncomputable section

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (-2 - (2 * Real.sqrt 5 / 5) * t, 2 + (2 * Real.sqrt 5 / 5) * t)

-- Define the distance function between a point and the line l
def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + 2 * y - 2) / Real.sqrt 5

-- Statement of the theorem
theorem min_distance_curve_to_line :
  let P := curve_C (π / 12)
  let min_dist := (2 * Real.sqrt 2 - 2) / Real.sqrt 5
  (∀ θ : ℝ, π / 12 < θ ∧ θ < π / 4 →
    distance_to_line (curve_C θ).1 (curve_C θ).2 ≥ min_dist) ∧
  distance_to_line P.1 P.2 = min_dist ∧
  P.1 = (Real.sqrt 6 + Real.sqrt 2) / 2 ∧
  P.2 = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l426_42608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_count_l426_42647

theorem article_count (cost_price selling_price : ℝ) (gain_percent : ℝ) : 
  gain_percent = 11.11111111111111 →
  50 * cost_price = 45 * selling_price →
  selling_price = cost_price * (1 + gain_percent / 100) →
  45 = 50 * (100 / (100 + gain_percent)) := by
  intro h1 h2 h3
  -- Proof steps would go here
  sorry

#check article_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_count_l426_42647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_l426_42685

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem stating that g(x) = g⁻¹(x) when x = 3
theorem g_equals_g_inv_at_three :
  ∃! x : ℝ, g x = g_inv x ∧ x = 3 := by
  sorry

#check g_equals_g_inv_at_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_l426_42685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l426_42646

theorem ticket_price_possibilities : 
  let possible_prices := {x : ℕ | x > 0 ∧ 72 % x = 0 ∧ 120 % x = 0}
  Finset.card (Finset.filter (λ x => x > 0 ∧ 72 % x = 0 ∧ 120 % x = 0) (Finset.range 121)) = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l426_42646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l426_42641

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc 1 3 ∧
  (∀ (x : ℝ), x ∈ Set.Icc 1 3 → y x ≤ y x_max) ∧
  y x_max = 13 ∧
  x_max = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l426_42641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_stops_at_start_l426_42601

/-- Represents a circular track with quarters A, B, C, and D -/
inductive Quarter
| A
| B
| C
| D

/-- Represents a point on the circular track -/
structure Point where
  quarter : Quarter
  distance : ℚ  -- Distance from the start of the quarter, using rational numbers

/-- The circumference of the track in feet -/
def track_circumference : ℚ := 40

/-- The total distance run by Kim in feet -/
def total_distance : ℚ := 2000

/-- The starting point, which is at the beginning of quarter A -/
def start_point : Point := ⟨Quarter.A, 0⟩

/-- Calculate the ending point after running a given distance -/
noncomputable def end_point (distance : ℚ) : Point :=
  let laps := (distance / track_circumference).floor
  let remaining_distance := distance - (laps * track_circumference)
  ⟨Quarter.A, remaining_distance⟩

theorem kim_stops_at_start :
  end_point total_distance = start_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_stops_at_start_l426_42601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l426_42684

theorem min_value_expression (a b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, y > 0 → (Real.exp x / 2 - Real.log (2 * y))^2 + (x - y)^2 ≥ 2 * (1 - Real.log 2)^2) ∧
  (∃ x y : ℝ, y > 0 ∧ (Real.exp x / 2 - Real.log (2 * y))^2 + (x - y)^2 = 2 * (1 - Real.log 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l426_42684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_l426_42676

-- Define the quadrilateral ABCD and point E
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the area function
noncomputable def area (p q r : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the area function for quadrilaterals
noncomputable def area_quad (p q r s : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the line segment
def LineSegment (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Given conditions
axiom intersect_at_E : E ∈ LineSegment A C ∧ E ∈ LineSegment B D
axiom area_ABE : area A B E = 40
axiom area_ADE : area A D E = 25

-- Theorem to prove
theorem area_ABCD : area_quad A B C D = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_l426_42676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_for_odd_monotonic_function_l426_42600

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then Real.exp x + a 
  else if x < 0 then -(Real.exp (-x) + a)
  else 0

-- State the theorem
theorem min_value_of_a_for_odd_monotonic_function :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -(f a (-x))) →  -- f is odd
  (∀ x : ℝ, x > 0 → f a x = Real.exp x + a) →  -- f(x) = e^x + a for x > 0
  (∀ x y : ℝ, x < y → f a x < f a y) →  -- f is strictly increasing (monotonic)
  a ≥ -1 ∧ ∀ b : ℝ, b < -1 → ¬(∀ x y : ℝ, x < y → f b x < f b y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_for_odd_monotonic_function_l426_42600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_16pi_over_5_l426_42681

/-- Line L with parametric equation x = 1 + t, y = 4 - 2t -/
def L : ℝ → ℝ × ℝ := λ t => (1 + t, 4 - 2*t)

/-- Circle C with equation (x-2)² + y² = 4 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

/-- The intersection points of L and C -/
def intersection : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t, L t = p ∧ p ∈ C}

/-- The area of the circle with diameter AB, where A and B are in the intersection -/
noncomputable def area_circle_AB : ℝ := 
  let AB := Real.sqrt (((L 0).1 - (L 1).1)^2 + ((L 0).2 - (L 1).2)^2)
  Real.pi * (AB / 2)^2

/-- The main theorem -/
theorem area_is_16pi_over_5 : area_circle_AB = 16 * Real.pi / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_16pi_over_5_l426_42681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l426_42619

/-- The time (in seconds) required for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 130 meters long traveling at 54 km/hr takes 30 seconds to cross a 320-meter bridge -/
theorem train_crossing_bridge :
  train_crossing_time 130 54 320 = 30 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l426_42619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l426_42656

def Triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_sum_range (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  c = 2 →
  Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin A * Real.sin B = Real.sin C ^ 2 →
  2 < a + b ∧ a + b ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l426_42656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_count_l426_42673

theorem fox_count (total_animals cows sheep : ℕ) (fox_count : ℕ → ℕ) :
  total_animals = 100 →
  cows = 20 →
  sheep = 20 →
  (λ (f : ℕ) => cows + sheep + f + 3 * f) = fox_count →
  fox_count total_animals = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_count_l426_42673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_product_l426_42690

theorem max_value_sin_cos_product (x y z : ℝ) :
  Real.sin x * Real.sin y * Real.sin z + Real.cos x * Real.cos y * Real.cos z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_product_l426_42690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_difference_l426_42665

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => sequence_a (n + 2) * (sequence_a (n + 1) / sequence_a (n + 2) + 1)

theorem sequence_a_difference : sequence_a 6 - sequence_a 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_difference_l426_42665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l426_42677

-- Define the function (noncomputable due to log function)
noncomputable def f (x : ℝ) : ℝ := (x + 1)^0 + Real.log (-x^2 - 3*x + 4)

-- Define the domain
def domain : Set ℝ := {x | -4 < x ∧ x < -1 ∨ -1 < x ∧ x < 1}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l426_42677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_equals_one_l426_42606

/-- The function f(x) = x³ - ax² + x has a tangent line at x = 1 that is parallel to y = 2x. -/
theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 + x
  (∃ b : ℝ, ∀ x : ℝ, (deriv f) 1 * (x - 1) + f 1 = 2*x + b) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_implies_a_equals_one_l426_42606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_spending_amount_l426_42699

def worker_earnings (day : ℕ) : ℚ :=
  if day % 2 = 1 then 24 else 0

def worker_spendings (day : ℕ) (S : ℚ) : ℚ :=
  if day % 2 = 0 then S else 0

def net_total (n : ℕ) (S : ℚ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => worker_earnings (i + 1) - worker_spendings (i + 1) S)

theorem worker_spending_amount :
  ∃ S : ℚ, 
    (∀ n : ℕ, n < 9 → net_total n S < 48) ∧
    net_total 9 S = 48 →
    S = 16 := by
  sorry

#eval net_total 9 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_spending_amount_l426_42699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_area_ratio_l426_42612

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a hexagon -/
structure Hexagon where
  A : Point
  L : Point
  M : Point
  C : Point
  J : Point
  I : Point

/-- The area of a shape -/
noncomputable def area (shape : Type) : ℝ := sorry

/-- Given three squares with equal areas, points C and D as midpoints of sides GH and HE respectively,
    and M as the midpoint of AB, prove that the ratio of the area of hexagon ALMCJI to the sum of
    the areas of the three squares is 1/3 -/
theorem hexagon_to_squares_area_ratio
  (square1 square2 square3 : Square)
  (hex : Hexagon)
  (h1 : area Square = area Square)
  (h2 : area Square = area Square)
  (h3 : hex.C.x = (square2.B.x + square2.C.x) / 2)
  (h4 : hex.C.y = (square2.B.y + square2.C.y) / 2)
  (h5 : hex.M.x = (square1.A.x + square1.B.x) / 2)
  (h6 : hex.M.y = (square1.A.y + square1.B.y) / 2) :
  area Hexagon / (area Square + area Square + area Square) = 1 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_area_ratio_l426_42612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l426_42617

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  downstream : ℝ
  upstream : ℝ
  current : ℝ

/-- Calculates the rower's speed in still water given their downstream and upstream speeds and the current speed -/
noncomputable def stillWaterSpeed (s : RowerSpeed) : ℝ :=
  (s.downstream + s.upstream) / 2

/-- Theorem stating that given the specific conditions, the rower's speed in still water is 15.5 kmph -/
theorem rower_still_water_speed (s : RowerSpeed) 
  (h1 : s.downstream = 24) 
  (h2 : s.upstream = 7) 
  (h3 : s.current = 8.5) : 
  stillWaterSpeed s = 15.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_still_water_speed_l426_42617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_sixth_sufficient_not_necessary_for_sin_half_l426_42674

theorem alpha_pi_sixth_sufficient_not_necessary_for_sin_half :
  (∃ α : Real, α ≠ π / 6 ∧ Real.sin α = 1 / 2) ∧
  (∀ α : Real, α = π / 6 → Real.sin α = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_sixth_sufficient_not_necessary_for_sin_half_l426_42674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_result_l426_42616

def triangle_theorem (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.cos C + c * Real.cos B = Real.sqrt 2 * a * Real.cos A ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_theorem_result (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_theorem a b c A B C) :
  A = Real.pi / 4 ∧ 
  (a = 5 ∧ b = 3 * Real.sqrt 2 → Real.cos (2 * C) = -24 / 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_result_l426_42616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equation_solution_l426_42642

theorem tangent_equation_solution (x : ℝ) :
  (Real.cos (3 * x) ≠ 0 ∧ Real.cos (2 * x) ≠ 0) →
  (3 * Real.tan (3 * x) - 4 * Real.tan (2 * x) = Real.tan (2 * x)^2 * Real.tan (3 * x) ↔
   ∃ k : ℤ, x = k * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equation_solution_l426_42642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_equals_twenty_l426_42649

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 6 = 10
  product_property : a 4 * a 8 = 45

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem stating that S₅ = 20 for the given arithmetic sequence -/
theorem sum_five_equals_twenty (seq : ArithmeticSequence) : sum_n seq 5 = 20 := by
  sorry

#check sum_five_equals_twenty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_equals_twenty_l426_42649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l426_42643

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) - 2 * (Real.cos x) ^ 2

-- State the theorem for the smallest positive period and maximum value
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∃ (M : ℝ), M = -1/2 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/6) → f x ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l426_42643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_difference_l426_42626

def company_a_price : ℝ := 4
def company_b_price : ℝ := 3.5
def company_a_quantity : ℕ := 300
def company_b_quantity : ℕ := 350

theorem earnings_difference :
  company_b_price * (company_b_quantity : ℝ) - company_a_price * (company_a_quantity : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_difference_l426_42626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_obtuse_l426_42630

theorem triangle_is_obtuse (A B C : ℝ) (h : ∃ (k : ℝ), Real.sin A = 5 * k ∧ Real.sin B = 12 * k ∧ Real.sin C = 14 * k) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a = 5 * b / 12 ∧ c = 7 * b / 6 ∧
  (a^2 + b^2 - c^2) / (2 * a * b) < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_obtuse_l426_42630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_413_in_decimal_l426_42625

theorem smallest_n_with_413_in_decimal : ∃ (n : ℕ), 
  (∀ (k : ℕ), k < n → 
    ¬∃ (m : ℕ), m < k ∧ 
    Nat.Coprime m k ∧ 
    ∃ (x y : ℕ), (1000 * m = 413 * k + 1000 * x + 413 * y) ∧ y < 1000) ∧
  (∃ (m : ℕ), m < n ∧ 
    Nat.Coprime m n ∧ 
    ∃ (x y : ℕ), (1000 * m = 413 * n + 1000 * x + 413 * y) ∧ y < 1000) ∧
  n = 414 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_413_in_decimal_l426_42625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_size_calculation_l426_42623

theorem family_size_calculation (fathers_side : ℕ) (mothers_side_percentage : ℚ) : 
  fathers_side = 20 →
  mothers_side_percentage = 150 / 100 →
  fathers_side + (fathers_side * mothers_side_percentage).floor = 50 := by
  intros h1 h2
  sorry

#eval (20 : ℕ) + ((20 : ℕ) * (150 / 100 : ℚ)).floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_size_calculation_l426_42623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l426_42689

/-- Represents a quadrilateral divided by its diagonals -/
structure DiagonalDividedQuadrilateral where
  /-- The areas of the four triangles formed by the diagonals -/
  triangle_areas : Fin 4 → ℝ
  /-- Three of the triangle areas are 10, 20, and 30 -/
  known_areas : ∃ (a b c : Fin 4) (x y z : ℝ),
    x ∈ ({10, 20, 30} : Set ℝ) ∧
    y ∈ ({10, 20, 30} : Set ℝ) ∧
    z ∈ ({10, 20, 30} : Set ℝ) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    triangle_areas a = x ∧
    triangle_areas b = y ∧
    triangle_areas c = z
  /-- The fourth triangle has an area greater than each of the other three -/
  fourth_largest : ∃ d : Fin 4, ∀ i : Fin 4, i ≠ d → triangle_areas d > triangle_areas i

/-- The total area of the quadrilateral is the sum of its four triangle areas -/
def total_area (q : DiagonalDividedQuadrilateral) : ℝ :=
  Finset.sum Finset.univ (fun i => q.triangle_areas i)

/-- Theorem: The area of the quadrilateral is 120 -/
theorem quadrilateral_area (q : DiagonalDividedQuadrilateral) : total_area q = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l426_42689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_N_on_circle_l426_42614

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the vertices and focal points
noncomputable def A₁ : ℝ × ℝ := (-2, 0)
noncomputable def A₂ : ℝ × ℝ := (2, 0)
noncomputable def F₁ : ℝ × ℝ := (-1, 0)
noncomputable def F₂ : ℝ × ℝ := (1, 0)

-- Define the focal distance and eccentricity
noncomputable def focal_distance : ℝ := 2
noncomputable def eccentricity : ℝ := 1/2

-- Define line l₁
def l₁ (x : ℝ) : Prop := x = -2

-- Define the circle
def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 + x - 2 = 0

-- Main theorem
theorem point_N_on_circle 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (P : ℝ × ℝ) 
  (hP : ellipse P.1 P.2 a b) :
  ∃ N : ℝ × ℝ, fixed_circle N.1 N.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_N_on_circle_l426_42614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l426_42697

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/4) + f (x + 3*Real.pi/4)

theorem problem_solution :
  (f (Real.pi/2) = 1) ∧
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧ p = 2*Real.pi) ∧
  (∀ x : ℝ, g x ≥ -2) ∧ (∃ x : ℝ, g x = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l426_42697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l426_42632

/-- Calculates the total percent decrease of a card's value over two years,
    given the percent decreases for each year. -/
noncomputable def totalPercentDecrease (firstYearDecrease secondYearDecrease : ℝ) : ℝ :=
  100 - (100 - firstYearDecrease) * (100 - secondYearDecrease) / 100

/-- Theorem stating that a 60% decrease followed by a 10% decrease
    results in a total 64% decrease. -/
theorem card_value_decrease : totalPercentDecrease 60 10 = 64 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l426_42632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_value_l426_42680

noncomputable section

-- Define the polynomial Q(x)
def Q (x d e : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + 9

-- Define the mean of zeros
def mean_of_zeros (d e : ℝ) : ℝ := -d / 9

-- Define the product of zeros
def product_of_zeros : ℝ := -3

-- Define the sum of coefficients
def sum_of_coefficients (d e : ℝ) : ℝ := 3 + d + e + 9

-- Theorem statement
theorem e_value (d e : ℝ) :
  mean_of_zeros d e = product_of_zeros ∧
  product_of_zeros = sum_of_coefficients d e →
  e = -42 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_value_l426_42680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_square_dissection_l426_42620

-- Define a polygon
structure Polygon where
  vertices : List (ℝ × ℝ)

-- Define congruence for polygons
def CongruentPolygons (p1 p2 : Polygon) : Prop := sorry

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a square
structure Square where
  side : ℝ

-- Define dissection of a shape into polygons
def Dissection (shape : Type) (polygons : List Polygon) : Prop := sorry

-- Helper function to check if a polygon is a rectangle
def IsRectangle (p : Polygon) : Prop := sorry

-- Theorem statement
theorem rectangle_and_square_dissection :
  ∃ (r : Rectangle) (s : Square) (polygons : List Polygon),
    Dissection Rectangle polygons ∧
    Dissection Square polygons ∧
    polygons.length = 15 ∧
    (∀ p ∈ polygons, ∀ q ∈ polygons, CongruentPolygons p q) ∧
    (∀ p ∈ polygons, ¬IsRectangle p) := by
  sorry

#check rectangle_and_square_dissection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_square_dissection_l426_42620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_example_l426_42602

/-- Given a normal trip distance and duration, and an actual trip duration,
    calculate the additional distance traveled. -/
noncomputable def additional_distance (normal_distance : ℝ) (normal_duration : ℝ) (actual_duration : ℝ) : ℝ :=
  (actual_duration / normal_duration * normal_distance) - normal_distance

/-- Theorem: For a normal trip of 150 miles taking 3 hours, and an actual trip taking 5 hours
    at the same speed, the additional distance traveled is 100 miles. -/
theorem additional_distance_example : additional_distance 150 3 5 = 100 := by
  unfold additional_distance
  -- The proof steps would go here, but we'll use 'sorry' for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_example_l426_42602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_heard_is_51_2_l426_42686

/-- Calculates the average number of minutes heard by an audience during a talk --/
noncomputable def averageMinutesHeard (totalMinutes : ℝ) (fullAudience percentFullTalk percentSlept percentHalfTalk : ℝ) : ℝ :=
  let percentThreeQuartersTalk := 1 - percentFullTalk - percentSlept - percentHalfTalk * (1 - percentFullTalk - percentSlept)
  let minutesHeardFullTalk := totalMinutes * percentFullTalk * fullAudience
  let minutesHeardHalfTalk := (totalMinutes / 2) * percentHalfTalk * (1 - percentFullTalk - percentSlept) * fullAudience
  let minutesHeardThreeQuartersTalk := (totalMinutes * 3 / 4) * percentThreeQuartersTalk * fullAudience
  (minutesHeardFullTalk + minutesHeardHalfTalk + minutesHeardThreeQuartersTalk) / fullAudience

/-- Theorem: The average number of minutes heard by the audience is 51.2 minutes --/
theorem average_minutes_heard_is_51_2 :
  averageMinutesHeard 80 100 0.25 0.15 0.4 = 51.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_heard_is_51_2_l426_42686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l426_42663

theorem chord_length (r d : ℝ) (hr : r = 6) (hd : d = 5) :
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l426_42663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l426_42672

/-- The eccentricity of a hyperbola whose asymptote intersects a specific parabola at only one point -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (∃! x : ℝ, (b / a) * x = x^2 + 1) → 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l426_42672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l426_42624

theorem ticket_price_possibilities : 
  let divisors := {x : ℕ | x > 0 ∧ 60 % x = 0 ∧ 90 % x = 0}
  Finset.card (Finset.filter (λ x => x > 0 ∧ 60 % x = 0 ∧ 90 % x = 0) (Finset.range 91)) = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l426_42624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lattice_points_count_l426_42603

/-- A lattice point is a point with integer coordinates -/
def LatticePoint (p : ℤ × ℤ) : Prop := True

/-- The set of lattice points on the hyperbola x^2 - y^2 = 2000^2 -/
def HyperbolaLatticePoints : Set (ℤ × ℤ) :=
  {p | p.1^2 - p.2^2 = 2000^2}

/-- The number of lattice points on the hyperbola x^2 - y^2 = 2000^2 is 98 -/
theorem hyperbola_lattice_points_count : 
  Finset.card (Finset.filter (fun p => p.1^2 - p.2^2 = 2000^2) (Finset.product (Finset.range 2001) (Finset.range 2001))) = 98 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lattice_points_count_l426_42603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l426_42667

theorem cube_root_equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ (3 - 1/x)^(1/3 : ℝ) = -4 ↔ x = 1/67 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l426_42667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_equation_l426_42607

/-- Triangle ABC with vertices A(-4,0), B(0,4), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry  -- The actual calculation of the circumcenter is omitted for simplicity

/-- The Euler line of a triangle -/
noncomputable def euler_line (t : Triangle) : ℝ → ℝ → Prop :=
  let g := centroid t
  let w := circumcenter t
  fun x y => (y - g.2) = ((w.2 - g.2) / (w.1 - g.1)) * (x - g.1)

/-- The specific triangle ABC from the problem -/
def triangle_ABC : Triangle where
  A := (-4, 0)
  B := (0, 4)
  C := (2, 0)

/-- Theorem: The Euler line of triangle ABC has the equation x - y + 2 = 0 -/
theorem euler_line_equation :
  ∀ x y, euler_line triangle_ABC x y ↔ x - y + 2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_equation_l426_42607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l426_42650

/-- Represents a rectangular box with length 2w, width w, and height h -/
structure Box (w h : ℝ) where
  length : ℝ := 2 * w
  width : ℝ := w
  height : ℝ := h

/-- Calculates the area of wrapping paper needed for a given box -/
def wrappingPaperArea (w h : ℝ) : ℝ :=
  2 * w^2 + 3 * w * h

/-- Theorem stating that the area of the wrapping paper for a box with dimensions 2w × w × h is 2w^2 + 3wh -/
theorem wrapping_paper_area_theorem (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  wrappingPaperArea w h = 2 * w^2 + 3 * w * h := by
  -- Unfold the definition of wrappingPaperArea
  unfold wrappingPaperArea
  -- The equality is now trivial
  rfl

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l426_42650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_top_coloring_l426_42628

-- Define the colors
inductive Color
| Red
| Green
| Blue
| Purple

-- Define a cube
structure Cube where
  bottom_corners : Fin 4 → Color
  top_corners : Fin 4 → Color

-- Define a valid coloring
def is_valid_coloring (c : Cube) : Prop :=
  -- All bottom corners are different colors
  (∀ i j : Fin 4, i ≠ j → c.bottom_corners i ≠ c.bottom_corners j) ∧
  -- All top corners are different colors
  (∀ i j : Fin 4, i ≠ j → c.top_corners i ≠ c.top_corners j) ∧
  -- Each vertical pair of corners has different colors
  (∀ i : Fin 4, c.bottom_corners i ≠ c.top_corners i)

-- The theorem to prove
theorem unique_top_coloring (bottom : Fin 4 → Color) :
  (∀ i j : Fin 4, i ≠ j → bottom i ≠ bottom j) →
  ∃! top : Fin 4 → Color, is_valid_coloring ⟨bottom, top⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_top_coloring_l426_42628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_six_divisors_sum_3500_l426_42605

/-- A natural number has exactly six divisors -/
def has_six_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

/-- The sum of divisors of a natural number -/
def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- The theorem stating that 1996 is the only natural number
    with exactly six divisors whose sum is 3500 -/
theorem unique_number_with_six_divisors_sum_3500 :
  ∀ n : ℕ, has_six_divisors n ∧ sum_of_divisors n = 3500 ↔ n = 1996 := by
  sorry

#check unique_number_with_six_divisors_sum_3500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_six_divisors_sum_3500_l426_42605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l426_42671

def old_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List Float) : Float :=
  (data.sum) / (data.length.toFloat)

def sample_variance (data : List Float) (mean : Float) : Float :=
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length.toFloat)

def x_bar : Float := sample_mean old_data
def y_bar : Float := sample_mean new_data
def s1_squared : Float := sample_variance old_data x_bar
def s2_squared : Float := sample_variance new_data y_bar

def significant_improvement (x_bar y_bar s1_squared s2_squared : Float) : Prop :=
  y_bar - x_bar ≥ 2 * (((s1_squared + s2_squared) / 10).sqrt)

theorem new_device_improvement :
  significant_improvement x_bar y_bar s1_squared s2_squared := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l426_42671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l426_42610

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x^2 - 13*x + 10) / (x - 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ (1 < x ∧ x < 2) ∨ x > 2}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 2 → (f x > 0 ↔ x ∈ solution_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l426_42610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_a_values_l426_42678

-- Define the parabola and its properties
def is_parabola (a : Real) : (Real × Real) → Prop :=
  fun p => p.1^2 = a * p.2

-- Define the directrix of the parabola
noncomputable def directrix (a : Real) : Real := -a / 4

-- Define the distance function between a point and a horizontal line
def distance_to_horizontal_line (p : Real × Real) (y : Real) : Real :=
  |p.2 - y|

-- The main theorem
theorem parabola_a_values :
  ∀ a : Real, 
    (distance_to_horizontal_line (0, 1) (directrix a) = 2) →
    (a = -12 ∨ a = 4) :=
by
  sorry

#check parabola_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_a_values_l426_42678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l426_42693

/-- Theorem about an acute triangle ABC with specific properties -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < Real.pi/2 ∧ 
  0 < B ∧ B < Real.pi/2 ∧ 
  0 < C ∧ C < Real.pi/2 ∧ 
  A + B + C = Real.pi ∧
  Real.sin C = 2 * Real.cos A * Real.sin (B + Real.pi/3) ∧
  b + c = 6 →
  A = Real.pi/3 ∧ 
  ∀ (AD : Real), AD ≤ 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l426_42693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_is_blue_l426_42682

/-- Represents the color of a sheep -/
inductive SheepColor
  | Blue
  | Red
  | Green

/-- Represents the state of the sheep population -/
structure SheepState where
  blue : Nat
  red : Nat
  green : Nat

/-- The initial state of the sheep population -/
def initialState : SheepState :=
  { blue := 22, red := 18, green := 15 }

/-- The total number of sheep -/
def totalSheep : Nat := initialState.blue + initialState.red + initialState.green

/-- Represents a meeting between two sheep of different colors -/
def meet (s : SheepState) (c1 c2 : SheepColor) : SheepState :=
  match c1, c2 with
  | SheepColor.Blue, SheepColor.Red => { s with blue := s.blue - 1, red := s.red - 1, green := s.green + 2 }
  | SheepColor.Blue, SheepColor.Green => { s with blue := s.blue - 1, red := s.red + 2, green := s.green - 1 }
  | SheepColor.Red, SheepColor.Blue => { s with blue := s.blue - 1, red := s.red - 1, green := s.green + 2 }
  | SheepColor.Red, SheepColor.Green => { s with blue := s.blue + 2, red := s.red - 1, green := s.green - 1 }
  | SheepColor.Green, SheepColor.Blue => { s with blue := s.blue - 1, red := s.red + 2, green := s.green - 1 }
  | SheepColor.Green, SheepColor.Red => { s with blue := s.blue + 2, red := s.red - 1, green := s.green - 1 }
  | _, _ => s  -- No change if same color

/-- Predicate to check if all sheep are the same color -/
def allSameColor (s : SheepState) : Prop :=
  (s.blue = totalSheep ∧ s.red = 0 ∧ s.green = 0) ∨
  (s.red = totalSheep ∧ s.blue = 0 ∧ s.green = 0) ∨
  (s.green = totalSheep ∧ s.blue = 0 ∧ s.red = 0)

/-- The main theorem to be proved -/
theorem final_state_is_blue (finalState : SheepState) :
  (∃ n : Nat, ∃ meetingSequence : List (SheepColor × SheepColor),
    (finalState = meetingSequence.foldl (fun s (c1, c2) => meet s c1 c2) initialState) ∧
    allSameColor finalState) →
  finalState.blue = totalSheep ∧ finalState.red = 0 ∧ finalState.green = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_is_blue_l426_42682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_colorings_l426_42640

/-- A coloring of integers is a function from ℤ to Bool, where true represents red and false represents blue. -/
def Coloring := ℤ → Bool

/-- A valid coloring satisfies two conditions:
    1. For all integers n, n and n+7 have the same color
    2. There does not exist an integer k such that k, k+1, and 2k all have the same color -/
def is_valid_coloring (c : Coloring) : Prop :=
  (∀ n : ℤ, c n = c (n + 7)) ∧
  (¬ ∃ k : ℤ, c k = c (k + 1) ∧ c k = c (2 * k))

/-- The number of valid colorings is exactly 6 -/
theorem num_valid_colorings :
  ∃ (s : Finset Coloring), (∀ c ∈ s, is_valid_coloring c) ∧ s.card = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_colorings_l426_42640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pu_length_l426_42657

/-- Triangle PQR with given side lengths -/
structure Triangle (P Q R : EuclideanSpace ℝ (Fin 2)) where
  pq : dist P Q = 13
  qr : dist Q R = 26
  rp : dist R P = 24

/-- Point S on QR is the intersection of angle bisector of ∠PQR with QR -/
noncomputable def S (P Q R : EuclideanSpace ℝ (Fin 2)) (tri : Triangle P Q R) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Point T is the intersection of angle bisector of ∠PQR with the circumcircle of PQR, T ≠ P -/
noncomputable def T (P Q R : EuclideanSpace ℝ (Fin 2)) (tri : Triangle P Q R) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Point U is the intersection of circumcircle of PST with PQ, U ≠ P -/
noncomputable def U (P Q R : EuclideanSpace ℝ (Fin 2)) (tri : Triangle P Q R) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Main theorem: PU = 26 -/
theorem pu_length (P Q R : EuclideanSpace ℝ (Fin 2)) (tri : Triangle P Q R) :
  dist P (U P Q R tri) = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pu_length_l426_42657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_inequality_l426_42661

theorem min_a_for_inequality : 
  (∀ x : ℝ, x ∈ Set.Icc 0 5 → |2 - x| + |x + 1| ≤ 9) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ |2 - x| + |x + 1| > 9 - ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_inequality_l426_42661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_approx_l426_42664

noncomputable def lawn_length : ℝ := 100
noncomputable def lawn_width : ℝ := 120
noncomputable def swath_width : ℝ := 30 / 12  -- Convert inches to feet
noncomputable def overlap : ℝ := 6 / 12  -- Convert inches to feet
noncomputable def mowing_speed : ℝ := 4500

noncomputable def effective_swath_width : ℝ := swath_width - overlap

noncomputable def num_strips : ℝ := lawn_width / effective_swath_width

noncomputable def total_distance : ℝ := num_strips * lawn_length

noncomputable def mowing_time : ℝ := total_distance / mowing_speed

theorem mowing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |mowing_time - 1.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_approx_l426_42664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l426_42633

noncomputable def f (a b x : ℝ) : ℝ := b * a^x

theorem exponential_function_properties 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a b 1 = 6) 
  (h4 : f a b 3 = 24) :
  (∀ x, f a b x = 3 * 2^x) ∧ 
  (∀ m, (∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) ↔ m ≤ 5/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l426_42633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_return_l426_42631

-- Define the track length in meters
noncomputable def track_length : ℝ := 400

-- Define the walking speed in meters per minute
noncomputable def speed_mpm : ℝ := 6000 / 60

-- Define the net distance walked
noncomputable def net_distance : ℝ := (1 - 3 + 5) * speed_mpm

-- Define the remaining distance to complete the circle
noncomputable def remaining_distance : ℝ := track_length - net_distance

-- Theorem statement
theorem min_time_to_return : remaining_distance / speed_mpm = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_return_l426_42631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_property_l426_42622

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the curve C (ellipse)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- Define the point T
def point_T : ℝ × ℝ := (4, 0)

-- Define the theorem
theorem trajectory_and_angle_property :
  ∃ (P : ℝ × ℝ → Prop),
    (∀ x y, P (x, y) ↔ 
      (∃ r > 0, 
        (∀ x' y', circle_M x' y' → ((x - x')^2 + (y - y')^2 = (r + 1)^2)) ∧
        (∀ x' y', circle_N x' y' → ((x - x')^2 + (y - y')^2 = (3 - r)^2)))) ∧
    (∀ x y, P (x, y) ↔ curve_C x y) ∧
    (∀ k : ℝ,
      ∀ R S : ℝ × ℝ,
        curve_C R.1 R.2 →
        curve_C S.1 S.2 →
        R.2 = k * (R.1 - 1) →
        S.2 = k * (S.1 - 1) →
        let O : ℝ × ℝ := (0, 0)
        let T : ℝ × ℝ := point_T
        (T.1 - O.1) * (R.2 - T.2) * (S.1 - T.1) =
        (T.1 - O.1) * (S.2 - T.2) * (R.1 - T.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_property_l426_42622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l426_42627

/-- The set of foci of a hyperbola -/
def foci (C : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Predicate indicating that a real number is the slope of an asymptote of a hyperbola -/
def is_asymptote_slope (C : Set (ℝ × ℝ)) (m : ℝ) : Prop := sorry

/-- Given two hyperbolas C₁ and C₂ with the following properties:
    1. The foci of C₁ coincide with those of C₂
    2. The equation of C₁ is x²/3 - y² = 1
    3. The slope of one asymptote of C₂ is twice the slope of one asymptote of C₁
    Then the equation of C₂ is x² - y²/3 = 1 -/
theorem hyperbola_equation (C₁ C₂ : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C₁ ↔ x^2/3 - y^2 = 1) →
  (∀ (f : ℝ × ℝ), f ∈ foci C₁ ↔ f ∈ foci C₂) →
  (∃ (m₁ m₂ : ℝ), is_asymptote_slope C₁ m₁ ∧ is_asymptote_slope C₂ m₂ ∧ m₂ = 2*m₁) →
  (∀ (x y : ℝ), (x, y) ∈ C₂ ↔ x^2 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l426_42627
