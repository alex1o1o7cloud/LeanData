import Mathlib

namespace rams_weight_increase_l2071_207170

theorem rams_weight_increase (ram_weight shyam_weight : ℝ) : 
  ram_weight / shyam_weight = 6 / 5 →
  ∃ (ram_increase : ℝ),
    ram_weight * (1 + ram_increase) + shyam_weight * 1.21 = 82.8 ∧
    (ram_weight * (1 + ram_increase) + shyam_weight * 1.21) / (ram_weight + shyam_weight) = 1.15 →
    ram_increase = 1.48 := by
  sorry

end rams_weight_increase_l2071_207170


namespace paige_mp3_songs_l2071_207169

theorem paige_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) : 
  initial = 11 → deleted = 9 → added = 8 → initial - deleted + added = 10 := by
  sorry

end paige_mp3_songs_l2071_207169


namespace stratified_sampling_results_l2071_207194

theorem stratified_sampling_results (junior_students senior_students sample_size : ℕ) 
  (h1 : junior_students = 400)
  (h2 : senior_students = 200)
  (h3 : sample_size = 60) :
  let junior_sample := (junior_students * sample_size) / (junior_students + senior_students)
  let senior_sample := sample_size - junior_sample
  Nat.choose junior_students junior_sample * Nat.choose senior_students senior_sample =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end stratified_sampling_results_l2071_207194


namespace geometric_sequence_a4_l2071_207105

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 3 - 34 * a 3 + 64 = 0 →
  a 5 * a 5 - 34 * a 5 + 64 = 0 →
  (a 4 = 8 ∨ a 4 = -8) :=
by sorry

end geometric_sequence_a4_l2071_207105


namespace intersection_distance_squared_example_l2071_207198

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculate the square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1 c2 : Circle) : ℝ :=
  let x1 := c1.center.1
  let y1 := c1.center.2
  let x2 := c2.center.1
  let y2 := c2.center.2
  let r1 := c1.radius
  let r2 := c2.radius
  -- Calculate the square of the distance between intersection points
  sorry

theorem intersection_distance_squared_example : 
  let c1 : Circle := ⟨(3, -2), 5⟩
  let c2 : Circle := ⟨(3, 4), Real.sqrt 13⟩
  intersection_distance_squared c1 c2 = 36 := by
  sorry

end intersection_distance_squared_example_l2071_207198


namespace book_pages_count_l2071_207136

/-- Calculates the total number of pages in a book given the number of pages read, left to read, and skipped. -/
def totalPages (pagesRead : ℕ) (pagesLeft : ℕ) (pagesSkipped : ℕ) : ℕ :=
  pagesRead + pagesLeft + pagesSkipped

/-- Proves that for the given numbers of pages read, left to read, and skipped, the total number of pages in the book is 372. -/
theorem book_pages_count : totalPages 125 231 16 = 372 := by
  sorry

end book_pages_count_l2071_207136


namespace line_length_calculation_line_length_proof_l2071_207108

theorem line_length_calculation (initial_length : ℕ) 
  (first_erasure : ℕ) (first_extension : ℕ) 
  (second_erasure : ℕ) (final_addition : ℕ) : ℕ :=
  let step1 := initial_length - first_erasure
  let step2 := step1 + first_extension
  let step3 := step2 - second_erasure
  let final_length := step3 + final_addition
  final_length

theorem line_length_proof :
  line_length_calculation 100 24 35 15 8 = 104 := by
  sorry

end line_length_calculation_line_length_proof_l2071_207108


namespace complex_power_72_l2071_207153

/-- Prove that (cos 215° + i sin 215°)^72 = 1 -/
theorem complex_power_72 : (Complex.exp (215 * Real.pi / 180 * Complex.I))^72 = 1 := by
  sorry

end complex_power_72_l2071_207153


namespace g_range_and_density_l2071_207123

noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

theorem g_range_and_density :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1 < g a b c ∧ g a b c < 2) ∧
  (∀ ε : ℝ, ε > 0 → 
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 1| < ε) ∧
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 2| < ε)) :=
by sorry

end g_range_and_density_l2071_207123


namespace reflection_property_l2071_207164

/-- A reflection in ℝ² -/
structure Reflection where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem stating that a reflection mapping (3, 2) to (1, 6) will map (2, -1) to (-2/5, -11/5) -/
theorem reflection_property (r : Reflection) 
  (h : r.apply (3, 2) = (1, 6)) :
  r.apply (2, -1) = (-2/5, -11/5) := by
  sorry

end reflection_property_l2071_207164


namespace sum_123_even_descending_l2071_207132

/-- The sum of the first n even natural numbers in descending order -/
def sumEvenDescending (n : ℕ) : ℕ :=
  n * (2 * n + 2) / 2

/-- Theorem: The sum of the first 123 even natural numbers in descending order is 15252 -/
theorem sum_123_even_descending : sumEvenDescending 123 = 15252 := by
  sorry

end sum_123_even_descending_l2071_207132


namespace divisibility_theorem_l2071_207187

theorem divisibility_theorem (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end divisibility_theorem_l2071_207187


namespace invalid_vote_percentage_l2071_207189

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℝ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 0.85)
  (h3 : candidate_a_votes = 404600) :
  (1 - (candidate_a_votes : ℝ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
  sorry

end invalid_vote_percentage_l2071_207189


namespace no_perfect_square_in_range_l2071_207173

theorem no_perfect_square_in_range : 
  ∀ n : ℕ, 4 ≤ n ∧ n ≤ 12 → ¬∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end no_perfect_square_in_range_l2071_207173


namespace complex_exponential_sum_l2071_207197

theorem complex_exponential_sum (γ δ : ℝ) : 
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -5/8 + 9/10 * Complex.I → 
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -5/8 - 9/10 * Complex.I := by
sorry

end complex_exponential_sum_l2071_207197


namespace division_problem_l2071_207172

theorem division_problem : ∃ (q : ℕ), 
  220070 = (555 + 445) * q + 70 ∧ q = 2 * (555 - 445) := by
  sorry

end division_problem_l2071_207172


namespace function_analysis_l2071_207141

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem function_analysis (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((2 / x) - (2 * a / x^3)) x) →
  (HasDerivAt (f a) 0 1 → a = 1) ∧
  (a > 0 → IsLocalMin (f a) (Real.sqrt a)) ∧
  ((∃ x y, 1 ≤ x ∧ x < y ∧ f a x = 2 ∧ f a y = 2) → 2 ≤ a ∧ a < Real.exp 1) :=
by sorry

end function_analysis_l2071_207141


namespace quadratic_solution_implies_a_minus_b_l2071_207190

theorem quadratic_solution_implies_a_minus_b (a b : ℝ) : 
  (4^2 + 4*a - 4*b = 0) → (a - b = -4) := by
  sorry

end quadratic_solution_implies_a_minus_b_l2071_207190


namespace range_of_m_l2071_207158

def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (¬p m ∧ q m) → (1 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l2071_207158


namespace equation_represents_hyperbola_l2071_207140

/-- The equation represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0) ∧
    (∀ x y : ℝ, x^2 - 36*y^2 - 12*x + y + 64 = 0 ↔ 
      a*(x - c)^2 + b*(y - d)^2 + e*(x - c) + f*(y - d) = 1) :=
sorry

end equation_represents_hyperbola_l2071_207140


namespace band_members_minimum_l2071_207100

theorem band_members_minimum (n : ℕ) : n = 165 ↔ 
  n > 0 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 7 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m % 9 = 7 → n ≤ m :=
by sorry

end band_members_minimum_l2071_207100


namespace probability_neither_orange_nor_white_l2071_207165

theorem probability_neither_orange_nor_white (orange black white : ℕ) 
  (h_orange : orange = 8) (h_black : black = 7) (h_white : white = 6) :
  (black : ℚ) / (orange + black + white) = 1 / 3 := by
sorry

end probability_neither_orange_nor_white_l2071_207165


namespace min_value_of_f_l2071_207160

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y a ≤ f x a) ∧
  (f x a ≤ 11) ∧
  (∃ z ∈ Set.Icc (-2 : ℝ) 2, f z a = 11) →
  (∃ w ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f w a ≤ f y a ∧ f w a = -29) :=
by sorry

end min_value_of_f_l2071_207160


namespace intersection_complement_equals_specific_set_l2071_207129

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equals_specific_set :
  B ∩ (U \ A) = {6, 7} := by
  sorry

end intersection_complement_equals_specific_set_l2071_207129


namespace local_odd_function_part1_local_odd_function_part2_l2071_207122

-- Definition of local odd function
def is_local_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, f (-x) = -f x

-- Part 1
theorem local_odd_function_part1 (m : ℝ) :
  is_local_odd_function (fun x => 2^x + m) (Set.Icc (-2) 2) →
  m ∈ Set.Icc (-17/8) (-1) :=
sorry

-- Part 2
theorem local_odd_function_part2 (m : ℝ) :
  is_local_odd_function (fun x => 4^x + m*2^(x+1) + m^2 - 4) Set.univ →
  m ∈ Set.Icc (-1) (Real.sqrt 10) :=
sorry

end local_odd_function_part1_local_odd_function_part2_l2071_207122


namespace short_walk_probability_l2071_207183

/-- The number of gates in the airport --/
def numGates : ℕ := 20

/-- The distance between adjacent gates in feet --/
def gateDistance : ℕ := 50

/-- The maximum distance Dave can walk in feet --/
def maxWalkDistance : ℕ := 200

/-- The probability of selecting two different gates that are at most maxWalkDistance apart --/
def probabilityShortWalk : ℚ := 67 / 190

theorem short_walk_probability :
  (numGates : ℚ) * (numGates - 1) * probabilityShortWalk =
    (2 * 4) +  -- Gates at extreme ends
    (6 * 5) +  -- Gates 2 to 4 and 17 to 19
    (12 * 8)   -- Gates 5 to 16
  ∧ maxWalkDistance / gateDistance = 4 := by sorry

end short_walk_probability_l2071_207183


namespace rational_equation_solution_l2071_207177

theorem rational_equation_solution (x : ℝ) : 
  x ≠ 3 → x ≠ -3 → (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by sorry

end rational_equation_solution_l2071_207177


namespace smallest_c_for_inverse_l2071_207152

def g (x : ℝ) := (x - 3)^2 + 6

theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end smallest_c_for_inverse_l2071_207152


namespace shopping_time_calculation_l2071_207137

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_waiting_time : ℕ
  total_active_shopping_time : ℕ
  total_trip_time : ℕ

/-- Calculates the time spent shopping and performing tasks -/
def time_shopping_and_tasks (trip : ShoppingTrip) : ℕ :=
  trip.total_trip_time - trip.total_waiting_time

/-- Theorem stating that the time spent shopping and performing tasks
    is equal to the total trip time minus the total waiting time -/
theorem shopping_time_calculation (trip : ShoppingTrip) 
  (h1 : trip.total_waiting_time = 58)
  (h2 : trip.total_active_shopping_time = 29)
  (h3 : trip.total_trip_time = 135) :
  time_shopping_and_tasks trip = 77 := by
  sorry

#eval time_shopping_and_tasks { total_waiting_time := 58, total_active_shopping_time := 29, total_trip_time := 135 }

end shopping_time_calculation_l2071_207137


namespace expression_evaluation_l2071_207107

theorem expression_evaluation : -3^2 + (-12) * |-(1/2)| - 6 / (-1) = -9 := by
  sorry

end expression_evaluation_l2071_207107


namespace triangle_third_angle_l2071_207199

theorem triangle_third_angle (a b c : ℝ) (ha : a = 70) (hb : b = 50) 
  (sum_of_angles : a + b + c = 180) : c = 60 := by
  sorry

end triangle_third_angle_l2071_207199


namespace inequality_equivalence_l2071_207149

theorem inequality_equivalence (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := by sorry

end inequality_equivalence_l2071_207149


namespace holly_insulin_pills_l2071_207147

/-- Represents the number of pills Holly takes per day for each type of medication -/
structure DailyPills where
  insulin : ℕ
  blood_pressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_total (d : DailyPills) : ℕ :=
  7 * (d.insulin + d.blood_pressure + d.anticonvulsant)

/-- Holly's daily pill regimen satisfies the given conditions -/
def holly_pills : DailyPills :=
  { insulin := 2,
    blood_pressure := 3,
    anticonvulsant := 6 }

theorem holly_insulin_pills :
  holly_pills.insulin = 2 ∧
  holly_pills.blood_pressure = 3 ∧
  holly_pills.anticonvulsant = 2 * holly_pills.blood_pressure ∧
  weekly_total holly_pills = 77 := by
  sorry

end holly_insulin_pills_l2071_207147


namespace donut_distribution_l2071_207196

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories,
    with at least one object in each category. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 35 ways to distribute 8 donuts into 5 varieties
    with at least one donut of each variety. -/
theorem donut_distribution : distribute 8 5 = 35 := by
  sorry

end donut_distribution_l2071_207196


namespace max_third_altitude_l2071_207163

/-- A scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  /-- The length of the first altitude -/
  altitude1 : ℝ
  /-- The length of the second altitude -/
  altitude2 : ℝ
  /-- The length of the third altitude -/
  altitude3 : ℝ
  /-- Condition that the triangle is scalene -/
  scalene : altitude1 ≠ altitude2 ∧ altitude2 ≠ altitude3 ∧ altitude3 ≠ altitude1
  /-- Condition that two altitudes are 6 and 18 -/
  specific_altitudes : (altitude1 = 6 ∧ altitude2 = 18) ∨ (altitude1 = 18 ∧ altitude2 = 6) ∨
                       (altitude1 = 6 ∧ altitude3 = 18) ∨ (altitude1 = 18 ∧ altitude3 = 6) ∨
                       (altitude2 = 6 ∧ altitude3 = 18) ∨ (altitude2 = 18 ∧ altitude3 = 6)
  /-- Condition that the third altitude is an integer -/
  integer_altitude : ∃ n : ℤ, (altitude3 : ℝ) = n ∨ (altitude2 : ℝ) = n ∨ (altitude1 : ℝ) = n

/-- The maximum possible integer length of the third altitude is 8 -/
theorem max_third_altitude (t : ScaleneTriangle) : 
  ∃ (max_altitude : ℕ), max_altitude = 8 ∧ 
  ∀ (n : ℕ), (n : ℝ) = t.altitude1 ∨ (n : ℝ) = t.altitude2 ∨ (n : ℝ) = t.altitude3 → n ≤ max_altitude :=
sorry

end max_third_altitude_l2071_207163


namespace problem_solution_l2071_207114

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 7)
  (h_eq2 : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end problem_solution_l2071_207114


namespace muffin_banana_price_ratio_l2071_207103

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  muffin_price > 0 →
  banana_price > 0 →
  4 * muffin_price + 3 * banana_price > 0 →
  2 * (4 * muffin_price + 3 * banana_price) = 2 * muffin_price + 16 * banana_price →
  muffin_price / banana_price = 5 / 3 := by
sorry

end muffin_banana_price_ratio_l2071_207103


namespace hexagon_interior_exterior_angle_sum_l2071_207157

theorem hexagon_interior_exterior_angle_sum : 
  ∃! n : ℕ, n > 2 ∧ (n - 2) * 180 = 2 * 360 := by sorry

#check hexagon_interior_exterior_angle_sum

end hexagon_interior_exterior_angle_sum_l2071_207157


namespace lattice_triangle_area_bound_l2071_207138

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define a lattice triangle
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

-- Function to count interior lattice points
def countInteriorLatticePoints (t : LatticeTriangle) : ℕ := sorry

-- Function to calculate the area of a triangle
def triangleArea (t : LatticeTriangle) : ℚ := sorry

-- Theorem statement
theorem lattice_triangle_area_bound 
  (t : LatticeTriangle) 
  (h : countInteriorLatticePoints t = 1) : 
  triangleArea t ≤ 9/2 := by sorry

end lattice_triangle_area_bound_l2071_207138


namespace largest_common_term_under_1000_l2071_207111

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 7 + 11 * m

/-- A common term of both sequences -/
def commonTerm (n m : ℕ) : Prop := seq1 n = seq2 m

/-- The largest common term less than 1000 -/
theorem largest_common_term_under_1000 :
  ∃ (n m : ℕ), commonTerm n m ∧ seq1 n = 974 ∧ 
  (∀ (k l : ℕ), commonTerm k l → seq1 k < 1000 → seq1 k ≤ 974) :=
sorry

end largest_common_term_under_1000_l2071_207111


namespace increasing_function_sum_implication_l2071_207120

theorem increasing_function_sum_implication (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y) :
  f a + f b > f (-a) + f (-b) → a + b > 0 := by
  sorry

end increasing_function_sum_implication_l2071_207120


namespace proportion_solution_l2071_207133

theorem proportion_solution (y : ℝ) : 
  (0.75 : ℝ) / 1.2 = y / 8 → y = 5 := by
sorry

end proportion_solution_l2071_207133


namespace geometric_sequence_sum_l2071_207154

def is_geometric_sequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : is_geometric_sequence a 6) 
  (h1 : a 1 = 1) (h2 : a 2 = 2) : 
  (Finset.range 9).sum (λ i => a (i + 1)) = 18 := by
  sorry

end geometric_sequence_sum_l2071_207154


namespace area_of_problem_l_shape_l2071_207175

/-- Represents an L-shaped figure with given dimensions -/
structure LShape where
  short_width : ℝ
  short_length : ℝ
  long_width : ℝ
  long_length : ℝ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shape (l : LShape) : ℝ :=
  l.short_width * l.short_length + l.long_width * l.long_length

/-- The specific L-shape from the problem -/
def problem_l_shape : LShape :=
  { short_width := 2
    short_length := 3
    long_width := 5
    long_length := 8 }

/-- Theorem stating that the area of the given L-shape is 46 square units -/
theorem area_of_problem_l_shape :
  area_of_l_shape problem_l_shape = 46 := by
  sorry


end area_of_problem_l_shape_l2071_207175


namespace greatest_integer_cube_root_three_l2071_207124

theorem greatest_integer_cube_root_three : ⌊(2 + Real.sqrt 3)^3⌋ = 51 := by
  sorry

end greatest_integer_cube_root_three_l2071_207124


namespace prism_on_sphere_surface_area_l2071_207168

/-- A right prism with all vertices on a sphere -/
structure PrismOnSphere where
  /-- The height of the prism -/
  height : ℝ
  /-- The volume of the prism -/
  volume : ℝ
  /-- The surface area of the sphere -/
  sphereSurfaceArea : ℝ

/-- Theorem: If a right prism with all vertices on a sphere has height 4 and volume 64,
    then the surface area of the sphere is 48π -/
theorem prism_on_sphere_surface_area (p : PrismOnSphere) 
    (h_height : p.height = 4)
    (h_volume : p.volume = 64) :
    p.sphereSurfaceArea = 48 * Real.pi := by
  sorry

end prism_on_sphere_surface_area_l2071_207168


namespace perpendicular_implies_parallel_l2071_207109

/-- Three lines in a plane -/
structure PlaneLine where
  dir : ℝ × ℝ  -- Direction vector

/-- Perpendicular lines -/
def perpendicular (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.1 + l1.dir.2 * l2.dir.2 = 0

/-- Parallel lines -/
def parallel (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.2 = l1.dir.2 * l2.dir.1

theorem perpendicular_implies_parallel (a b c : PlaneLine) :
  perpendicular a b → perpendicular b c → parallel a c := by
  sorry

end perpendicular_implies_parallel_l2071_207109


namespace range_of_a_l2071_207117

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -1/4 :=
by sorry

end range_of_a_l2071_207117


namespace break_even_is_80_weeks_l2071_207135

/-- Represents the chicken and egg problem --/
structure ChickenEggProblem where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken : ℕ
  eggs_bought_weekly : ℕ
  egg_cost_per_dozen : ℚ

/-- Calculates the break-even point in weeks --/
def break_even_weeks (problem : ChickenEggProblem) : ℕ :=
  sorry

/-- Theorem stating that the break-even point is 80 weeks for the given problem --/
theorem break_even_is_80_weeks (problem : ChickenEggProblem)
  (h1 : problem.num_chickens = 4)
  (h2 : problem.chicken_cost = 20)
  (h3 : problem.weekly_feed_cost = 1)
  (h4 : problem.eggs_per_chicken = 3)
  (h5 : problem.eggs_bought_weekly = 12)
  (h6 : problem.egg_cost_per_dozen = 2) :
  break_even_weeks problem = 80 :=
sorry

end break_even_is_80_weeks_l2071_207135


namespace line_point_value_l2071_207146

/-- Given a line passing through points (-1, y) and (4, k), with slope equal to k and k = 1, 
    prove that y = -4 -/
theorem line_point_value (y k : ℝ) (h1 : k = 1) 
    (h2 : (k - y) / (4 - (-1)) = k) : y = -4 := by
  sorry

end line_point_value_l2071_207146


namespace radius_of_touching_sphere_l2071_207127

/-- A regular quadrilateral pyramid with an inscribed sphere and a touching sphere -/
structure PyramidWithSpheres where
  -- Base side length
  a : ℝ
  -- Lateral edge length
  b : ℝ
  -- Radius of inscribed sphere Q₁
  r₁ : ℝ
  -- Radius of touching sphere Q₂
  r₂ : ℝ
  -- Condition: The pyramid is regular quadrilateral
  regular : a > 0 ∧ b > 0
  -- Condition: Q₁ is inscribed in the pyramid
  q₁_inscribed : r₁ > 0
  -- Condition: Q₂ touches Q₁ and all lateral faces
  q₂_touches : r₂ > 0

/-- Theorem stating the radius of Q₂ in the given pyramid configuration -/
theorem radius_of_touching_sphere (p : PyramidWithSpheres) 
  (h₁ : p.a = 6) 
  (h₂ : p.b = 5) : 
  p.r₂ = 3 * Real.sqrt 7 / 49 := by
  sorry


end radius_of_touching_sphere_l2071_207127


namespace hyperbola_trajectory_l2071_207144

/-- The trajectory of point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4,0) and F₂(4,0) -/
theorem hyperbola_trajectory (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (-4, 0)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (x, y)
  Real.sqrt ((x - 4)^2 + y^2) - Real.sqrt ((x + 4)^2 + y^2) = 4 →
  x^2 / 4 - y^2 / 12 = 1 ∧ x ≤ -2 :=
by sorry

end hyperbola_trajectory_l2071_207144


namespace sum_of_largest_odd_factors_l2071_207130

/-- The largest odd factor of a natural number -/
def largest_odd_factor (n : ℕ) : ℕ := sorry

/-- The sum of the first n terms of the sequence of largest odd factors -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem: The sum of the first 2^2016 - 1 terms of the sequence
    of largest odd factors is equal to (4^2016 - 1) / 3 -/
theorem sum_of_largest_odd_factors :
  S (2^2016 - 1) = (4^2016 - 1) / 3 := by sorry

end sum_of_largest_odd_factors_l2071_207130


namespace altitude_to_longest_side_l2071_207148

theorem altitude_to_longest_side (a b c h : ℝ) : 
  a = 8 → b = 15 → c = 17 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (1/2 * a * b) → 
  h = 120/17 := by
sorry

end altitude_to_longest_side_l2071_207148


namespace polynomial_simplification_l2071_207185

theorem polynomial_simplification (x : ℝ) : 
  5 - 5*x - 10*x^2 + 10 + 15*x - 20*x^2 - 10 + 20*x + 30*x^2 = 5 + 30*x := by
sorry

end polynomial_simplification_l2071_207185


namespace at_least_one_positive_solution_l2071_207162

def f (x : ℝ) : ℝ := x^10 + 4*x^9 + 7*x^8 + 2023*x^7 - 2024*x^6

theorem at_least_one_positive_solution :
  ∃ x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end at_least_one_positive_solution_l2071_207162


namespace floor_ceiling_sum_l2071_207156

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end floor_ceiling_sum_l2071_207156


namespace one_different_value_l2071_207151

/-- The standard exponentiation result for 2^(2^(2^2)) -/
def standard_result : ℕ := 65536

/-- The set of all possible values obtained by different parenthesizations of 2^2^2^2 -/
def all_results : Set ℕ :=
  {2^(2^(2^2)), 2^((2^2)^2), ((2^2)^2)^2, (2^(2^2))^2, (2^2)^(2^2)}

/-- The theorem stating that there is exactly one value different from the standard result -/
theorem one_different_value :
  ∃! x, x ∈ all_results ∧ x ≠ standard_result :=
sorry

end one_different_value_l2071_207151


namespace second_class_sample_size_l2071_207192

/-- Calculates the number of items to be sampled from a specific class in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (classPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (classPopulation * sampleSize) / totalPopulation

theorem second_class_sample_size :
  let totalPopulation : ℕ := 200
  let secondClassPopulation : ℕ := 60
  let sampleSize : ℕ := 40
  stratifiedSampleSize totalPopulation secondClassPopulation sampleSize = 12 := by
sorry

end second_class_sample_size_l2071_207192


namespace alan_wings_increase_l2071_207174

/-- Proves that Alan needs to increase his rate by 4 wings per minute to beat Kevin's record -/
theorem alan_wings_increase (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) : 
  kevin_wings = 64 → 
  kevin_time = 8 → 
  alan_rate = 5 → 
  (kevin_wings / kevin_time : ℚ) - alan_rate = 4 := by
  sorry

#check alan_wings_increase

end alan_wings_increase_l2071_207174


namespace total_roses_is_109_l2071_207180

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_is_109 : total_roses_needed = 109 := by
  sorry

end total_roses_is_109_l2071_207180


namespace factorization_proof_l2071_207176

theorem factorization_proof (a b c : ℝ) :
  4 * a * b + 2 * a * c = 2 * a * (2 * b + c) := by
  sorry

end factorization_proof_l2071_207176


namespace move_point_on_number_line_l2071_207186

theorem move_point_on_number_line (start : ℤ) (movement : ℤ) (result : ℤ) :
  start = -2 →
  movement = 3 →
  result = start + movement →
  result = 1 := by
  sorry

end move_point_on_number_line_l2071_207186


namespace tan_double_angle_l2071_207195

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l2071_207195


namespace inequality_implies_a_bound_l2071_207115

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end inequality_implies_a_bound_l2071_207115


namespace problem_solution_l2071_207167

def f (x : ℝ) : ℝ := x^2 + 10

def g (x : ℝ) : ℝ := x^2 - 6

theorem problem_solution (a : ℝ) (h1 : a > 3) (h2 : f (g a) = 16) : a = Real.sqrt (Real.sqrt 6 + 6) := by
  sorry

end problem_solution_l2071_207167


namespace basketball_team_lineups_l2071_207110

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid lineups for the basketball team -/
def validLineups : ℕ :=
  choose 15 7 - choose 13 5

theorem basketball_team_lineups :
  validLineups = 5148 := by sorry

end basketball_team_lineups_l2071_207110


namespace quadratic_transformation_l2071_207118

/-- Given a quadratic function ax² + bx + c that can be expressed as 5(x - 5)² - 3,
    prove that when 4ax² + 4bx + 4c is expressed as n(x - h)² + k, h = 5. -/
theorem quadratic_transformation (a b c : ℝ) : 
  (∃ x, ax^2 + b*x + c = 5*(x - 5)^2 - 3) → 
  (∃ n k, ∀ x, 4*a*x^2 + 4*b*x + 4*c = n*(x - 5)^2 + k) := by
  sorry

end quadratic_transformation_l2071_207118


namespace sequence_max_value_l2071_207193

/-- The sequence a_n defined by -2n^2 + 29n + 3 for positive integers n has a maximum value of 108 -/
theorem sequence_max_value :
  ∃ (M : ℕ), ∀ (n : ℕ), n > 0 → (-2 * n^2 + 29 * n + 3 : ℤ) ≤ M ∧
  ∃ (k : ℕ), k > 0 ∧ (-2 * k^2 + 29 * k + 3 : ℤ) = M ∧ M = 108 :=
by
  sorry


end sequence_max_value_l2071_207193


namespace line_plane_parallelism_l2071_207101

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (l m : Line) (α : Plane) :
  subset m α → 
  parallel_lines l m → 
  ¬subset l α → 
  parallel_line_plane l α :=
sorry

end line_plane_parallelism_l2071_207101


namespace decimal_number_calculation_l2071_207171

theorem decimal_number_calculation (A B : ℝ) 
  (h1 : B - A = 211.5)
  (h2 : B = 10 * A) : 
  A = 23.5 := by
sorry

end decimal_number_calculation_l2071_207171


namespace intersection_A_B_l2071_207125

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end intersection_A_B_l2071_207125


namespace equation_solution_l2071_207116

theorem equation_solution (x : ℝ) (h : x ≥ 0) : x + 2 * Real.sqrt x - 8 = 0 ↔ x = 4 := by
  sorry

end equation_solution_l2071_207116


namespace negation_of_existence_sqrt_leq_x_minus_one_negation_l2071_207126

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) :=
by sorry

end negation_of_existence_sqrt_leq_x_minus_one_negation_l2071_207126


namespace quarters_percentage_is_fifty_percent_l2071_207128

/-- The number of dimes -/
def num_dimes : ℕ := 50

/-- The number of quarters -/
def num_quarters : ℕ := 20

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- Theorem stating that the percentage of the total value in quarters is 50% -/
theorem quarters_percentage_is_fifty_percent :
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 50 := by sorry

end quarters_percentage_is_fifty_percent_l2071_207128


namespace max_value_implies_m_value_l2071_207134

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem max_value_implies_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-2) 2, f m x = 3) →
  m = 3 :=
sorry

end max_value_implies_m_value_l2071_207134


namespace fibonacci_pair_characterization_l2071_207142

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci_pair (a b : ℝ) : Prop :=
  ∀ n, ∃ m, a * (fibonacci n) + b * (fibonacci (n + 1)) = fibonacci m

theorem fibonacci_pair_characterization :
  ∀ a b : ℝ, is_fibonacci_pair a b ↔ 
    ((a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) ∨ 
     (∃ k : ℕ, a = fibonacci k ∧ b = fibonacci (k + 1))) :=
sorry

end fibonacci_pair_characterization_l2071_207142


namespace two_special_numbers_l2071_207181

/-- A three-digit number divisible by 5 that can be represented as n^3 + n^2 -/
def special_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧  -- three-digit number
  x % 5 = 0 ∧           -- divisible by 5
  ∃ n : ℕ, x = n^3 + n^2  -- can be represented as n^3 + n^2

/-- There are exactly two numbers satisfying the special_number property -/
theorem two_special_numbers : ∃! (a b : ℕ), a ≠ b ∧ special_number a ∧ special_number b ∧
  ∀ x, special_number x → (x = a ∨ x = b) :=
by sorry

end two_special_numbers_l2071_207181


namespace equal_intercept_line_equations_l2071_207182

/-- A line with equal absolute intercepts passing through (3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, -2)
  point_condition : -2 = m * 3 + b
  -- The line has equal absolute intercepts
  intercept_condition : ∃ (k : ℝ), k ≠ 0 ∧ (k = -b/m ∨ k = b)

/-- The possible equations of the line -/
def possible_equations (l : EqualInterceptLine) : Prop :=
  (2 * l.m + 3 = 0 ∧ l.b = 0) ∨
  (l.m = -1 ∧ l.b = 1) ∨
  (l.m = 1 ∧ l.b = 5)

theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine), possible_equations l :=
sorry

end equal_intercept_line_equations_l2071_207182


namespace inequality_holds_l2071_207112

theorem inequality_holds (a : ℝ) : (∀ x > 1, x^2 + a*x - 6 > 0) ↔ a ≥ 5 := by sorry

end inequality_holds_l2071_207112


namespace inequality_proof_l2071_207179

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) :
  a * d > b * c := by
  sorry

end inequality_proof_l2071_207179


namespace quadratic_two_roots_condition_l2071_207145

theorem quadratic_two_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - 1 + m = 0 ∧ y^2 + 2*y - 1 + m = 0) ↔ m ≤ 2 := by
  sorry

end quadratic_two_roots_condition_l2071_207145


namespace modular_product_equivalence_l2071_207178

theorem modular_product_equivalence (n : ℕ) : 
  (507 * 873) % 77 = n ∧ 0 ≤ n ∧ n < 77 → n = 15 := by
  sorry

end modular_product_equivalence_l2071_207178


namespace flag_arrangement_remainder_l2071_207155

/-- The number of ways to arrange flags on two poles -/
def M : ℕ := sorry

/-- The total number of flags -/
def total_flags : ℕ := 24

/-- The number of blue flags -/
def blue_flags : ℕ := 14

/-- The number of red flags -/
def red_flags : ℕ := 10

/-- Each flagpole has at least one flag -/
axiom at_least_one_flag : M > 0

/-- Each sequence starts with a blue flag -/
axiom starts_with_blue : True

/-- No two red flags on either pole are adjacent -/
axiom no_adjacent_red : True

theorem flag_arrangement_remainder :
  M % 1000 = 1 := by sorry

end flag_arrangement_remainder_l2071_207155


namespace norma_cards_l2071_207166

theorem norma_cards (initial_cards : ℕ) (lost_fraction : ℚ) (remaining_cards : ℕ) : 
  initial_cards = 88 → 
  lost_fraction = 3/4 → 
  remaining_cards = initial_cards - (initial_cards * lost_fraction).floor → 
  remaining_cards = 22 := by
sorry

end norma_cards_l2071_207166


namespace amount_after_two_years_l2071_207150

theorem amount_after_two_years 
  (initial_amount : ℝ) 
  (annual_increase_rate : ℝ) 
  (years : ℕ) :
  initial_amount = 32000 →
  annual_increase_rate = 1 / 8 →
  years = 2 →
  initial_amount * (1 + annual_increase_rate) ^ years = 40500 := by
sorry

end amount_after_two_years_l2071_207150


namespace min_value_problem_l2071_207191

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_eq : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*(y'^2) ≥ min :=
sorry

end min_value_problem_l2071_207191


namespace simplify_sqrt_product_l2071_207159

theorem simplify_sqrt_product : Real.sqrt 18 * Real.sqrt 72 = 12 * Real.sqrt 2 := by sorry

end simplify_sqrt_product_l2071_207159


namespace proportion_solution_l2071_207106

theorem proportion_solution (x : ℝ) (h : (0.75 : ℝ) / x = 5 / 8) : x = 1.2 := by
  sorry

end proportion_solution_l2071_207106


namespace sum_distances_constant_l2071_207113

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the triangle -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- A point inside an equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) where
  /-- The distance from the point to the first side -/
  dist1 : ℝ
  /-- The distance from the point to the second side -/
  dist2 : ℝ
  /-- The distance from the point to the third side -/
  dist3 : ℝ
  /-- Assumption that all distances are non-negative -/
  dist_nonneg : dist1 ≥ 0 ∧ dist2 ≥ 0 ∧ dist3 ≥ 0
  /-- Assumption that the point is inside the triangle -/
  inside : dist1 + dist2 + dist3 < t.side_length * Real.sqrt 3 / 2

/-- The theorem stating that the sum of distances is constant -/
theorem sum_distances_constant (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.dist1 + p.dist2 + p.dist3 = t.side_length * Real.sqrt 3 / 2 := by
  sorry

end sum_distances_constant_l2071_207113


namespace pupils_in_program_l2071_207143

/-- Given a program with a total of 238 people and 61 parents, prove that there were 177 pupils present. -/
theorem pupils_in_program (total_people : ℕ) (parents : ℕ) (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end pupils_in_program_l2071_207143


namespace triangle_ratio_theorem_l2071_207139

theorem triangle_ratio_theorem (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (b + c) / (Real.sin B + Real.sin C) = 2 :=
by sorry

end triangle_ratio_theorem_l2071_207139


namespace max_sum_squares_roots_l2071_207188

/-- 
For a quadratic equation x^2 + 2ax + 2a^2 + 4a + 3 = 0 with parameter a,
the sum of squares of its roots is maximized when a = -3, and the maximum sum is 18.
-/
theorem max_sum_squares_roots (a : ℝ) : 
  let f := fun x : ℝ => x^2 + 2*a*x + 2*a^2 + 4*a + 3
  let sum_squares := (- (2*a))^2 - 2*(2*a^2 + 4*a + 3)
  (∀ b : ℝ, sum_squares ≤ (-8 * (-3) - 6)) ∧ 
  sum_squares = 18 ↔ a = -3 := by sorry

end max_sum_squares_roots_l2071_207188


namespace y_coordinate_of_P_l2071_207131

/-- The y-coordinate of point P given specific conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-3, 2) →
  C = (3, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = 6/7 := by
  sorry

#check y_coordinate_of_P

end y_coordinate_of_P_l2071_207131


namespace jennys_pen_cost_l2071_207184

/-- Proves that the cost of each pen is $1.50 given the conditions of Jenny's purchase --/
theorem jennys_pen_cost 
  (print_cost : ℚ) 
  (copies : ℕ) 
  (pages : ℕ) 
  (num_pens : ℕ) 
  (payment : ℚ) 
  (change : ℚ)
  (h1 : print_cost = 1 / 10)
  (h2 : copies = 7)
  (h3 : pages = 25)
  (h4 : num_pens = 7)
  (h5 : payment = 40)
  (h6 : change = 12) :
  (payment - change - print_cost * copies * pages) / num_pens = 3 / 2 := by
  sorry

end jennys_pen_cost_l2071_207184


namespace no_integer_solutions_l2071_207121

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2022 + y^2 = 2*y + 2 := by
  sorry

end no_integer_solutions_l2071_207121


namespace max_product_sum_320_l2071_207161

theorem max_product_sum_320 : 
  ∃ (a b : ℤ), a + b = 320 ∧ 
  ∀ (x y : ℤ), x + y = 320 → x * y ≤ a * b ∧
  a * b = 25600 := by
sorry

end max_product_sum_320_l2071_207161


namespace mathathon_problem_count_l2071_207119

theorem mathathon_problem_count (rounds : Nat) (problems_per_round : Nat) : 
  rounds = 7 → problems_per_round = 3 → rounds * problems_per_round = 21 := by
  sorry

end mathathon_problem_count_l2071_207119


namespace twelve_people_round_table_l2071_207102

/-- The number of distinct seating arrangements for n people around a round table,
    where rotations are considered the same. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct seating arrangements for 12 people around a round table,
    where rotations are considered the same, is equal to 11!. -/
theorem twelve_people_round_table : roundTableArrangements 12 = 39916800 := by
  sorry

end twelve_people_round_table_l2071_207102


namespace theater_ticket_cost_l2071_207104

/-- Proves that the cost of an adult ticket is $7.56 given the conditions of the theater ticket sales. -/
theorem theater_ticket_cost (total_tickets : ℕ) (total_receipts : ℚ) (adult_tickets : ℕ) (child_ticket_cost : ℚ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 90 →
  child_ticket_cost = 4 →
  ∃ adult_ticket_cost : ℚ,
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 756 / 100 := by
  sorry

end theater_ticket_cost_l2071_207104
