import Mathlib

namespace max_candies_for_one_student_l1368_136850

theorem max_candies_for_one_student
  (num_students : ℕ)
  (mean_candies : ℕ)
  (h_num_students : num_students = 40)
  (h_mean_candies : mean_candies = 6)
  (h_at_least_one : ∀ student, student ≥ 1) :
  ∃ (max_candies : ℕ), max_candies = 201 ∧
    ∀ (student_candies : ℕ),
      student_candies ≤ max_candies ∧
      (num_students - 1) * 1 + student_candies ≤ num_students * mean_candies :=
by sorry

end max_candies_for_one_student_l1368_136850


namespace even_integer_sequence_sum_l1368_136802

theorem even_integer_sequence_sum : 
  ∀ (a b c d : ℤ),
  (∃ (k₁ k₂ k₃ k₄ : ℤ), a = 2 * k₁ ∧ b = 2 * k₂ ∧ c = 2 * k₃ ∧ d = 2 * k₄) →
  (0 < a) →
  (a < b) →
  (b < c) →
  (c < d) →
  (d - a = 90) →
  (∃ (r : ℤ), c - b = b - a ∧ b - a = r) →
  (∃ (q : ℚ), c / b = d / c ∧ c / b = q) →
  (a + b + c + d = 194) :=
by sorry

end even_integer_sequence_sum_l1368_136802


namespace number_puzzle_l1368_136885

theorem number_puzzle : ∃! x : ℝ, 150 - x = x + 68 :=
  sorry

end number_puzzle_l1368_136885


namespace linear_function_fits_points_l1368_136886

-- Define the set of points
def points : List (ℝ × ℝ) := [(0, 150), (1, 120), (2, 90), (3, 60), (4, 30)]

-- Define the linear function
def f (x : ℝ) : ℝ := -30 * x + 150

-- Theorem statement
theorem linear_function_fits_points : 
  ∀ (point : ℝ × ℝ), point ∈ points → f point.1 = point.2 := by
  sorry

end linear_function_fits_points_l1368_136886


namespace complex_product_theorem_l1368_136842

theorem complex_product_theorem : 
  let z : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 11))
  (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * 
  (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33) = 2197 := by
  sorry

end complex_product_theorem_l1368_136842


namespace badminton_players_count_l1368_136859

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  tennis_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculates the number of badminton players in the sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.tennis_players - club.both_players)

/-- Theorem stating that in a specific sports club configuration, 
    the number of badminton players is 20 -/
theorem badminton_players_count (club : SportsClub) 
  (h1 : club.total_members = 42)
  (h2 : club.tennis_players = 23)
  (h3 : club.neither_players = 6)
  (h4 : club.both_players = 7) :
  badminton_players club = 20 := by
  sorry

end badminton_players_count_l1368_136859


namespace valid_distributions_count_l1368_136878

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of days
def num_days : ℕ := 2

-- Function to calculate the number of valid distributions
def count_valid_distributions (students : ℕ) (days : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem valid_distributions_count :
  count_valid_distributions num_students num_days = 14 := by
  sorry

end valid_distributions_count_l1368_136878


namespace system_solutions_l1368_136833

def is_solution (x y z w : ℝ) : Prop :=
  x^2 + 2*y^2 + 2*z^2 + w^2 = 43 ∧
  y^2 + z^2 + w^2 = 29 ∧
  5*z^2 - 3*w^2 + 4*x*y + 12*y*z + 6*z*x = 95

theorem system_solutions :
  {(x, y, z, w) : ℝ × ℝ × ℝ × ℝ | is_solution x y z w} =
  {(1, 2, 3, 4), (1, 2, 3, -4), (-1, -2, -3, 4), (-1, -2, -3, -4)} :=
by sorry

end system_solutions_l1368_136833


namespace floor_sum_example_l1368_136837

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1368_136837


namespace angle_measure_in_special_triangle_l1368_136874

theorem angle_measure_in_special_triangle (A B C : ℝ) :
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  B = 2 * A →        -- ∠B is twice ∠A
  C = 4 * A →        -- ∠C is four times ∠A
  B = 360 / 7 :=     -- Measure of ∠B is 360/7°
by
  sorry

end angle_measure_in_special_triangle_l1368_136874


namespace inequality_condition_neither_sufficient_nor_necessary_l1368_136810

theorem inequality_condition_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, (a > b → 1/a < 1/b) → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (1/a < 1/b)) :=
sorry

end inequality_condition_neither_sufficient_nor_necessary_l1368_136810


namespace cube_geometric_shapes_l1368_136839

structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

def isRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isParallelogramNotRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isRightAngledTetrahedron (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def fourVertices (c : Cube) : Type :=
  {v : Fin 4 → Fin 8 // ∀ i j, i ≠ j → v i ≠ v j}

theorem cube_geometric_shapes (c : Cube) :
  ∃ (v : fourVertices c),
    isRectangle (fun i => c.vertices (v.val i)) ∧
    isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∧
    isRightAngledTetrahedron (fun i => c.vertices (v.val i)) ∧
    ¬∃ (w : fourVertices c),
      (isRectangle (fun i => c.vertices (w.val i)) ∧
       isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ∧
       isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ∧
       (isRectangle (fun i => c.vertices (w.val i)) ≠
        isRectangle (fun i => c.vertices (v.val i)) ∨
        isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ≠
        isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∨
        isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ≠
        isRightAngledTetrahedron (fun i => c.vertices (v.val i)))) :=
  sorry

end cube_geometric_shapes_l1368_136839


namespace exists_prime_not_dividing_euclid_l1368_136883

/-- Definition of Euclid numbers -/
def euclid : ℕ → ℕ
  | 0 => 3
  | n + 1 => euclid n * euclid (n - 1) + 1

/-- Theorem: There exists a prime that does not divide any Euclid number -/
theorem exists_prime_not_dividing_euclid : ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬(p ∣ euclid n) := by
  sorry

end exists_prime_not_dividing_euclid_l1368_136883


namespace inequality_proof_l1368_136891

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end inequality_proof_l1368_136891


namespace unique_solution_for_C_equality_l1368_136895

-- Define C(k) as the sum of distinct prime divisors of k
def C (k : ℕ+) : ℕ := sorry

-- Theorem statement
theorem unique_solution_for_C_equality :
  ∀ n : ℕ+, C (2^n.val + 1) = C n ↔ n = 3 := by sorry

end unique_solution_for_C_equality_l1368_136895


namespace compare_abc_l1368_136893

theorem compare_abc : ∃ (a b c : ℝ),
  a = 2 * Real.log 1.01 ∧
  b = Real.log 1.02 ∧
  c = Real.sqrt 1.04 - 1 ∧
  c < a ∧ a < b :=
by sorry

end compare_abc_l1368_136893


namespace three_five_power_sum_l1368_136828

theorem three_five_power_sum (x y : ℕ+) (h : 3^(x.val) * 5^(y.val) = 225) : x.val + y.val = 4 := by
  sorry

end three_five_power_sum_l1368_136828


namespace problem_solution_l1368_136868

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ a ↔ 1 ≤ a ∧ a ≤ 2) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → m + 2*n = 2 → 1/m + 1/n ≥ 3/2 + Real.sqrt 2) :=
by sorry

#check problem_solution

end problem_solution_l1368_136868


namespace integer_pair_divisibility_l1368_136872

theorem integer_pair_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a ^ 3 = k * b ^ 2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) →
  b = 1 ∨ b = a := by
sorry

end integer_pair_divisibility_l1368_136872


namespace tangent_slope_when_chord_maximized_l1368_136858

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) : Prop := circle_M P.1 P.2

-- Define a tangent line from a point on M to O
def is_tangent_line (P : ℝ × ℝ) (m : ℝ) : Prop :=
  point_on_M P ∧ ∃ A : ℝ × ℝ, circle_O A.1 A.2 ∧ (A.2 - P.2) = m * (A.1 - P.1)

-- Define the other intersection point Q
def other_intersection (P : ℝ × ℝ) (m : ℝ) (Q : ℝ × ℝ) : Prop :=
  point_on_M Q ∧ (Q.2 - P.2) = m * (Q.1 - P.1) ∧ P ≠ Q

-- Theorem statement
theorem tangent_slope_when_chord_maximized :
  ∃ P : ℝ × ℝ, ∃ m : ℝ, is_tangent_line P m ∧
  (∀ Q : ℝ × ℝ, other_intersection P m Q →
    ∀ P' : ℝ × ℝ, ∀ m' : ℝ, ∀ Q' : ℝ × ℝ,
      is_tangent_line P' m' ∧ other_intersection P' m' Q' →
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ (P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) →
  m = -7 ∨ m = 1 := by
sorry

end tangent_slope_when_chord_maximized_l1368_136858


namespace solve_iterated_f_equation_l1368_136836

def f (x : ℝ) : ℝ := x^2 + 6*x + 6

def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem solve_iterated_f_equation :
  ∃ x : ℝ, iterate_f 2017 x = 2017 ∧
  x = -3 + (2020 : ℝ)^(1/(2^2017)) ∨
  x = -3 - (2020 : ℝ)^(1/(2^2017)) :=
sorry

end solve_iterated_f_equation_l1368_136836


namespace diophantine_solutions_l1368_136879

/-- Theorem: Solutions for the Diophantine equations 3a + 5b = 1, 3a + 5b = 4, and 183a + 117b = 3 -/
theorem diophantine_solutions :
  (∀ (a b : ℤ), 3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k) ∧
  (∀ (a b : ℤ), 3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k) ∧
  (∀ (a b : ℤ), 183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k) :=
sorry

/-- Lemma: The solution set for 3a + 5b = 1 is correct -/
lemma solution_set_3a_5b_1 (a b : ℤ) :
  3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k :=
sorry

/-- Lemma: The solution set for 3a + 5b = 4 is correct -/
lemma solution_set_3a_5b_4 (a b : ℤ) :
  3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k :=
sorry

/-- Lemma: The solution set for 183a + 117b = 3 is correct -/
lemma solution_set_183a_117b_3 (a b : ℤ) :
  183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k :=
sorry

end diophantine_solutions_l1368_136879


namespace log_relationship_depends_on_base_l1368_136899

noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_relationship_depends_on_base (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (a1 a2 : ℝ), 
    (a1 > 0 ∧ a1 ≠ 1 ∧ log a1 2 + log a1 10 > 2 * log a1 6) ∧
    (a2 > 0 ∧ a2 ≠ 1 ∧ log a2 2 + log a2 10 < 2 * log a2 6) :=
by sorry

end log_relationship_depends_on_base_l1368_136899


namespace sqrt_negative_product_equals_two_sqrt_two_l1368_136820

theorem sqrt_negative_product_equals_two_sqrt_two :
  Real.sqrt ((-4) * (-2)) = 2 * Real.sqrt 2 := by sorry

end sqrt_negative_product_equals_two_sqrt_two_l1368_136820


namespace other_focus_coordinates_l1368_136818

/-- An ellipse with specific properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis at the origin -/
  tangent_at_origin : True
  /-- The length of the major axis -/
  major_axis_length : ℝ
  /-- The coordinates of one focus -/
  focus1 : ℝ × ℝ

/-- Theorem: Given an ellipse with specific properties, the other focus has coordinates (-3, -4) -/
theorem other_focus_coordinates (e : Ellipse) 
  (h1 : e.major_axis_length = 20)
  (h2 : e.focus1 = (3, 4)) :
  ∃ (other_focus : ℝ × ℝ), other_focus = (-3, -4) := by
  sorry

end other_focus_coordinates_l1368_136818


namespace distance_after_two_hours_l1368_136823

-- Define the walking rates and duration
def jay_rate : ℚ := 1 / 12  -- miles per minute
def sarah_rate : ℚ := 3 / 36  -- miles per minute
def duration : ℚ := 2 * 60  -- 2 hours in minutes

-- Define the theorem
theorem distance_after_two_hours :
  let jay_distance := jay_rate * duration
  let sarah_distance := sarah_rate * duration
  jay_distance + sarah_distance = 20 := by sorry

end distance_after_two_hours_l1368_136823


namespace sector_central_angle_l1368_136822

/-- Given a sector with radius r and perimeter 3r, its central angle is 1. -/
theorem sector_central_angle (r : ℝ) (h : r > 0) :
  (∃ (l : ℝ), l > 0 ∧ 2 * r + l = 3 * r) →
  (∃ (α : ℝ), α = l / r ∧ α = 1) :=
by sorry

end sector_central_angle_l1368_136822


namespace comic_books_liked_by_males_l1368_136890

theorem comic_books_liked_by_males 
  (total : ℕ) 
  (female_like_percent : ℚ) 
  (dislike_percent : ℚ) 
  (h_total : total = 300)
  (h_female_like : female_like_percent = 30 / 100)
  (h_dislike : dislike_percent = 30 / 100) :
  (total : ℚ) * (1 - female_like_percent - dislike_percent) = 120 := by
  sorry

end comic_books_liked_by_males_l1368_136890


namespace base5_polynomial_representation_l1368_136881

/-- Represents a polynomial in base 5 with coefficients less than 5 -/
def Base5Polynomial (coeffs : List Nat) : Prop :=
  coeffs.all (· < 5) ∧ coeffs.length > 0

/-- Converts a list of coefficients to a natural number in base 5 -/
def toBase5Number (coeffs : List Nat) : Nat :=
  coeffs.foldl (fun acc d => 5 * acc + d) 0

/-- Theorem: A polynomial with coefficients less than 5 can be uniquely represented as a base-5 number -/
theorem base5_polynomial_representation 
  (coeffs : List Nat) 
  (h : Base5Polynomial coeffs) :
  ∃! n : Nat, n = toBase5Number coeffs :=
sorry

end base5_polynomial_representation_l1368_136881


namespace max_handshakes_correct_l1368_136876

/-- The number of men shaking hands -/
def n : ℕ := 40

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- The maximum number of handshakes without cyclic handshakes -/
def maxHandshakes : ℕ := n.choose k

theorem max_handshakes_correct :
  maxHandshakes = 780 := by sorry

end max_handshakes_correct_l1368_136876


namespace hens_and_cows_l1368_136824

/-- Given a total number of animals and feet, calculates the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  total_animals - (total_feet - 2 * total_animals) / 2

/-- Theorem stating that given 50 animals with 140 feet, where hens have 2 feet
    and cows have 4 feet, the number of hens is 30 -/
theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 140) :
  number_of_hens total_animals total_feet = 30 := by
  sorry

#eval number_of_hens 50 140  -- Should output 30

end hens_and_cows_l1368_136824


namespace donation_in_scientific_notation_l1368_136830

/-- Definition of a billion in the context of this problem -/
def billion : ℕ := 10^8

/-- The donation amount in yuan -/
def donation : ℚ := 2.94 * billion

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℚ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that 2.94 billion yuan is equal to 2.94 × 10^8 in scientific notation -/
theorem donation_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℚ)^sn.exponent) = donation ∧
    sn.coefficient = 2.94 ∧ sn.exponent = 8 := by sorry

end donation_in_scientific_notation_l1368_136830


namespace number_of_rooms_l1368_136856

/-- Calculates the number of equal-sized rooms given the original dimensions,
    increase in dimensions, and total area. --/
def calculate_rooms (original_length original_width increase_dim total_area : ℕ) : ℕ :=
  let new_length := original_length + increase_dim
  let new_width := original_width + increase_dim
  let room_area := new_length * new_width
  let double_room_area := 2 * room_area
  let equal_rooms_area := total_area - double_room_area
  equal_rooms_area / room_area

/-- Theorem stating that the number of equal-sized rooms is 4 --/
theorem number_of_rooms : calculate_rooms 13 18 2 1800 = 4 := by
  sorry

end number_of_rooms_l1368_136856


namespace national_lipstick_day_attendance_l1368_136877

theorem national_lipstick_day_attendance (total_students : ℕ) : 
  (total_students : ℚ) / 2 = (total_students : ℚ) / 2 / 4 * 5 + 5 → total_students = 200 :=
by
  sorry

end national_lipstick_day_attendance_l1368_136877


namespace red_packs_count_l1368_136857

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := sorry

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

theorem red_packs_count : red_packs = 4 := by
  sorry

end red_packs_count_l1368_136857


namespace largest_number_from_hcf_lcm_factors_l1368_136816

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_ab : Nat.gcd a b = 50)
  (lcm_factor1 : ∃ k : ℕ+, Nat.lcm a b = 50 * 11 * k)
  (lcm_factor2 : ∃ k : ℕ+, Nat.lcm a b = 50 * 12 * k) :
  max a b = 600 := by
sorry

end largest_number_from_hcf_lcm_factors_l1368_136816


namespace no_discount_on_backpacks_l1368_136840

/-- Proves that there is no discount on the backpacks given the problem conditions -/
theorem no_discount_on_backpacks 
  (num_backpacks : ℕ) 
  (monogram_cost : ℚ) 
  (total_cost : ℚ) 
  (h1 : num_backpacks = 5)
  (h2 : monogram_cost = 12)
  (h3 : total_cost = 140) :
  total_cost = num_backpacks * monogram_cost + (total_cost - num_backpacks * monogram_cost) := by
  sorry

#check no_discount_on_backpacks

end no_discount_on_backpacks_l1368_136840


namespace withdraw_300_from_two_banks_in_20_bills_l1368_136864

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 20 = 30 := by
  sorry

end withdraw_300_from_two_banks_in_20_bills_l1368_136864


namespace angle_measure_in_triangle_l1368_136835

/-- Given a triangle XYZ where the measure of ∠X is 78 degrees and 
    the measure of ∠Y is 14 degrees less than four times the measure of ∠Z,
    prove that the measure of ∠Z is 23.2 degrees. -/
theorem angle_measure_in_triangle (X Y Z : ℝ) 
  (h1 : X = 78)
  (h2 : Y = 4 * Z - 14)
  (h3 : X + Y + Z = 180) :
  Z = 23.2 := by
  sorry

end angle_measure_in_triangle_l1368_136835


namespace range_of_m_given_one_root_l1368_136863

/-- The function f(x) defined in terms of x and m -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

/-- The property that f has exactly one root in [0, 1] -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc 0 1 ∧ f x m = 0

/-- The theorem stating the range of m given the condition -/
theorem range_of_m_given_one_root :
  ∀ m, has_one_root_in_unit_interval m → m ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
by sorry

end range_of_m_given_one_root_l1368_136863


namespace ellipse_minor_axis_length_l1368_136817

/-- The length of the minor axis of the ellipse 9x^2 + y^2 = 36 is 4 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} ∧
    2 * min a b = 4 :=
by sorry

end ellipse_minor_axis_length_l1368_136817


namespace family_savings_l1368_136880

def income : ℕ := 509600
def expenses : ℕ := 276000
def initial_savings : ℕ := 1147240

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end family_savings_l1368_136880


namespace area_of_inscribed_square_on_hypotenuse_l1368_136875

/-- An isosceles right triangle with inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the square inscribed with one side on a leg -/
  s : ℝ
  /-- Side length of the square inscribed with one side on the hypotenuse -/
  S : ℝ
  /-- The area of the square inscribed with one side on a leg is 484 -/
  h_area_s : s^2 = 484
  /-- Relationship between s and S in an isosceles right triangle -/
  h_relation : 3 * S = s * Real.sqrt 2

/-- 
Theorem: In an isosceles right triangle, if a square inscribed with one side on a leg 
has an area of 484 cm², then a square inscribed with one side on the hypotenuse 
has an area of 968/9 cm².
-/
theorem area_of_inscribed_square_on_hypotenuse 
  (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 968 / 9 := by
  sorry

end area_of_inscribed_square_on_hypotenuse_l1368_136875


namespace quadratic_equation_roots_range_l1368_136889

theorem quadratic_equation_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ 
    x^2 - a*x + a^2 - 4 = 0 ∧ 
    y^2 - a*y + a^2 - 4 = 0) ↔ 
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end quadratic_equation_roots_range_l1368_136889


namespace complex_equation_solution_l1368_136866

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (hi : i * i = -1) 
  (h : (1 + a * i) * i = 3 + i) : 
  a = -3 := by
sorry

end complex_equation_solution_l1368_136866


namespace quadratic_properties_l1368_136821

/-- A quadratic function f(x) = mx² + (m-2)x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

/-- The function is symmetric about the y-axis -/
def is_symmetric (m : ℝ) : Prop := ∀ x, f m x = f m (-x)

theorem quadratic_properties (m : ℝ) (h : is_symmetric m) :
  m = 2 ∧ 
  (∀ x y, x < y → f m x < f m y) ∧ 
  (∀ x, x > 0 → f m x > f m 0) ∧
  (∀ x, f m x ≥ 2) ∧ 
  f m 0 = 2 :=
sorry

end quadratic_properties_l1368_136821


namespace shopping_mall_problem_l1368_136853

/-- Represents the price of product A in yuan -/
def price_A : ℝ := 16

/-- Represents the price of product B in yuan -/
def price_B : ℝ := 4

/-- Represents the maximum number of product A that can be purchased -/
def max_A : ℕ := 41

theorem shopping_mall_problem :
  (20 * price_A + 15 * price_B = 380) ∧
  (15 * price_A + 10 * price_B = 280) ∧
  (∀ x : ℕ, x ≤ 100 → 
    (x * price_A + (100 - x) * price_B ≤ 900 → x ≤ max_A)) ∧
  (max_A * price_A + (100 - max_A) * price_B ≤ 900) :=
by sorry

end shopping_mall_problem_l1368_136853


namespace tablecloth_radius_l1368_136887

/-- Given a round tablecloth with a diameter of 10 feet, its radius is 5 feet. -/
theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 := by
  sorry

end tablecloth_radius_l1368_136887


namespace trig_sum_equals_sqrt_two_l1368_136873

theorem trig_sum_equals_sqrt_two : 
  Real.tan (60 * π / 180) + 2 * Real.sin (45 * π / 180) - 2 * Real.cos (30 * π / 180) = Real.sqrt 2 := by
  sorry

end trig_sum_equals_sqrt_two_l1368_136873


namespace house_price_proof_l1368_136804

theorem house_price_proof (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end house_price_proof_l1368_136804


namespace escalator_length_proof_l1368_136813

/-- The length of an escalator in feet. -/
def escalator_length : ℝ := 160

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 8

/-- The walking speed of a person on the escalator in feet per second. -/
def person_speed : ℝ := 2

/-- The time taken by the person to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 16

/-- Theorem stating that the length of the escalator is 160 feet, given the conditions. -/
theorem escalator_length_proof :
  escalator_length = (escalator_speed + person_speed) * time_taken :=
by sorry

end escalator_length_proof_l1368_136813


namespace systematic_sample_fourth_element_l1368_136884

/-- Represents a systematic sample of size 4 from 52 employees -/
structure SystematicSample where
  size : Nat
  total : Nat
  elements : Fin 4 → Nat
  is_valid : size = 4 ∧ total = 52 ∧ ∀ i, elements i ≤ total

/-- Checks if a given sample is arithmetic -/
def is_arithmetic_sample (s : SystematicSample) : Prop :=
  ∃ d, ∀ i j, s.elements i - s.elements j = (i.val - j.val : ℤ) * d

/-- The main theorem -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.elements 0 = 6)
  (h2 : s.elements 2 = 32)
  (h3 : s.elements 3 = 45)
  (h4 : is_arithmetic_sample s) :
  s.elements 1 = 19 := by sorry

end systematic_sample_fourth_element_l1368_136884


namespace phi_value_l1368_136882

theorem phi_value (φ : Real) (h1 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ)
  (h2 : 0 < φ ∧ φ < π / 2) : φ = 15 * π / 180 := by
  sorry

end phi_value_l1368_136882


namespace sinusoidal_function_omega_l1368_136847

/-- Given a sinusoidal function y = 2sin(ωx + π/6) with ω > 0,
    if the distance between adjacent symmetry axes is π/2,
    then ω = 2. -/
theorem sinusoidal_function_omega (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + π / (2 * ω)) + π / 6)) →
  ω = 2 := by
  sorry

end sinusoidal_function_omega_l1368_136847


namespace arithmetic_sequence_problem_l1368_136845

/-- An arithmetic sequence with positive common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 * seq.a 6 = 8 * seq.a 4)
  (h2 : seq.a 2 = 2) :
  seq.a 1 = 1 := by
  sorry

end arithmetic_sequence_problem_l1368_136845


namespace hyperbola_real_axis_length_l1368_136815

/-- The length of the real axis of the hyperbola 2x^2 - y^2 = 8 is 4 -/
theorem hyperbola_real_axis_length :
  ∃ (a : ℝ), a > 0 ∧ (∀ x y : ℝ, 2 * x^2 - y^2 = 8 → x^2 / (2 * a^2) - y^2 / (2 * a^2) = 1) ∧ 2 * a = 4 := by
  sorry

end hyperbola_real_axis_length_l1368_136815


namespace parabola_tangent_to_line_l1368_136862

/-- A parabola y = ax^2 + 6 is tangent to the line y = x if and only if a = 1/24 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 6 = x ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ y) ↔ a = 1/24 := by
  sorry

end parabola_tangent_to_line_l1368_136862


namespace equilibrium_portion_above_water_l1368_136860

/-- Represents a uniform rod partially submerged in water -/
structure PartiallySubmergedRod where
  /-- Length of the rod -/
  length : ℝ
  /-- Density of the rod -/
  density : ℝ
  /-- Density of water -/
  water_density : ℝ
  /-- Portion of the rod above water -/
  above_water_portion : ℝ

/-- Theorem stating the equilibrium condition for a partially submerged rod -/
theorem equilibrium_portion_above_water (rod : PartiallySubmergedRod)
  (h_positive_length : rod.length > 0)
  (h_density_ratio : rod.density = (5 / 9) * rod.water_density)
  (h_equilibrium : rod.above_water_portion * rod.length * rod.water_density * (rod.length / 2) =
                   (1 - rod.above_water_portion) * rod.length * rod.density * (rod.length / 2)) :
  rod.above_water_portion = 1 / 3 := by
  sorry

end equilibrium_portion_above_water_l1368_136860


namespace original_to_doubled_ratio_l1368_136869

theorem original_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 6) = 72 → x / (2 * x) = 1 / 2 := by
  sorry

end original_to_doubled_ratio_l1368_136869


namespace infinitely_many_perfect_squares_in_sequence_l1368_136851

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ x y : ℕ+, x > k ∧ y^2 = ⌊x * Real.sqrt 2⌋ := by
  sorry

end infinitely_many_perfect_squares_in_sequence_l1368_136851


namespace sequence_limit_existence_l1368_136826

theorem sequence_limit_existence (a : ℕ → ℝ) (h : ∀ n, 0 ≤ a n ∧ a n ≤ 1) :
  ∃ (n : ℕ → ℕ) (A : ℝ),
    (∀ i j, i < j → n i < n j) ∧
    (∀ ε > 0, ∃ N, ∀ i j, i ≠ j → i > N → j > N → |a (n i + n j) - A| < ε) := by
  sorry

end sequence_limit_existence_l1368_136826


namespace cylinder_surface_area_l1368_136846

/-- The total surface area of a right cylinder with height 8 and radius 3 is 66π -/
theorem cylinder_surface_area :
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 66 * π := by sorry

end cylinder_surface_area_l1368_136846


namespace lcm_factor_problem_l1368_136888

theorem lcm_factor_problem (A B : ℕ+) : 
  Nat.gcd A B = 10 →
  A = 150 →
  11 ∣ Nat.lcm A B →
  Nat.lcm A B = 10 * 11 * 15 := by
sorry

end lcm_factor_problem_l1368_136888


namespace transformation_result_l1368_136801

/-- Rotates a point (x, y) 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (1, -4) → b - a = -3 := by
  sorry

end transformation_result_l1368_136801


namespace rectangular_field_width_l1368_136827

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 :=
by
  sorry

end rectangular_field_width_l1368_136827


namespace sweet_cookies_eaten_l1368_136805

theorem sweet_cookies_eaten (initial_sweet : ℕ) (final_sweet : ℕ) (eaten_sweet : ℕ) :
  initial_sweet = final_sweet + eaten_sweet →
  eaten_sweet = initial_sweet - final_sweet :=
by sorry

end sweet_cookies_eaten_l1368_136805


namespace polynomial_evaluation_l1368_136843

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end polynomial_evaluation_l1368_136843


namespace event_occurrence_limit_l1368_136831

theorem event_occurrence_limit (ε : ℝ) (hε : 0 < ε) :
  ∀ δ > 0, ∃ N : ℕ, ∀ n ≥ N, 1 - (1 - ε)^n > 1 - δ :=
sorry

end event_occurrence_limit_l1368_136831


namespace probability_three_or_more_smile_l1368_136812

def probability_single_baby_smile : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_smile (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Finset.sum (Finset.range 3) (λ k => (n.choose k : ℚ) * p^k * (1 - p)^(n - k)))

theorem probability_three_or_more_smile :
  probability_at_least_three_smile probability_single_baby_smile number_of_babies = 353 / 729 :=
sorry

end probability_three_or_more_smile_l1368_136812


namespace cubic_fraction_equals_nine_l1368_136867

theorem cubic_fraction_equals_nine (x y : ℝ) (hx : x = 7) (hy : y = 2) :
  (x^3 + y^3) / (x^2 - x*y + y^2) = 9 := by
  sorry

end cubic_fraction_equals_nine_l1368_136867


namespace positive_decreasing_function_l1368_136849

/-- A function f: ℝ → ℝ is decreasing if for all x < y, f(x) > f(y) -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The second derivative of a function f: ℝ → ℝ -/
def SecondDerivative (f : ℝ → ℝ) : ℝ → ℝ := sorry

theorem positive_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : Decreasing f)
  (h_second_derivative : ∀ x, f x / SecondDerivative f x < 1 - x) :
  ∀ x, f x > 0 := by
  sorry

end positive_decreasing_function_l1368_136849


namespace empty_set_implies_a_zero_l1368_136819

theorem empty_set_implies_a_zero (a : ℝ) : (∀ x : ℝ, ax + 2 ≠ 0) → a = 0 := by
  sorry

end empty_set_implies_a_zero_l1368_136819


namespace sqrt_equation_sum_l1368_136838

theorem sqrt_equation_sum (y : ℝ) (d e f : ℕ+) : 
  y = Real.sqrt ((Real.sqrt 73 / 3) + (5 / 3)) →
  y^52 = 3*y^50 + 10*y^48 + 25*y^46 - y^26 + d*y^22 + e*y^20 + f*y^18 →
  d + e + f = 184 := by
  sorry

end sqrt_equation_sum_l1368_136838


namespace modulus_of_z_l1368_136809

-- Define the complex number z
def z : ℂ := Complex.I * (2 - Complex.I)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_z_l1368_136809


namespace sandra_current_age_l1368_136803

/-- Sandra's current age -/
def sandra_age : ℕ := 36

/-- Sandra's son's current age -/
def son_age : ℕ := 14

/-- Theorem stating Sandra's current age based on the given conditions -/
theorem sandra_current_age : 
  (sandra_age - 3 = 3 * (son_age - 3)) → sandra_age = 36 := by
  sorry

end sandra_current_age_l1368_136803


namespace negation_of_implication_l1368_136834

theorem negation_of_implication (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end negation_of_implication_l1368_136834


namespace expected_marbles_theorem_l1368_136806

/-- The expected number of marbles drawn until the special marble is picked, given that no ugly marbles were drawn -/
def expected_marbles_drawn (blue_marbles : ℕ) (ugly_marbles : ℕ) (special_marbles : ℕ) : ℚ :=
  let total_marbles := blue_marbles + ugly_marbles + special_marbles
  let prob_blue := blue_marbles / total_marbles
  let prob_special := special_marbles / total_marbles
  let expected_draws := (prob_blue / (1 - prob_blue)) / prob_special
  expected_draws

/-- Theorem stating that the expected number of marbles drawn is 20/11 -/
theorem expected_marbles_theorem :
  expected_marbles_drawn 9 10 1 = 20 / 11 := by
  sorry

end expected_marbles_theorem_l1368_136806


namespace arctan_sum_three_four_l1368_136865

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end arctan_sum_three_four_l1368_136865


namespace last_digit_of_N_l1368_136844

theorem last_digit_of_N (total_coins : ℕ) (h : total_coins = 3080) : 
  ∃ N : ℕ, (N * (N + 1)) / 2 = total_coins ∧ N % 10 = 8 := by
  sorry

end last_digit_of_N_l1368_136844


namespace average_age_of_three_students_l1368_136800

theorem average_age_of_three_students
  (total_students : ℕ)
  (total_average : ℝ)
  (eleven_students : ℕ)
  (eleven_average : ℝ)
  (fifteenth_student_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : eleven_students = 11)
  (h4 : eleven_average = 16)
  (h5 : fifteenth_student_age = 7)
  : (total_students * total_average - eleven_students * eleven_average - fifteenth_student_age) / (total_students - eleven_students - 1) = 14 := by
  sorry

end average_age_of_three_students_l1368_136800


namespace perimeter_difference_zero_l1368_136852

/-- Perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- Perimeter of Figure 1 --/
def figure1Perimeter : ℕ := rectanglePerimeter 6 1 + 4

/-- Perimeter of Figure 2 --/
def figure2Perimeter : ℕ := rectanglePerimeter 7 2

theorem perimeter_difference_zero : figure1Perimeter = figure2Perimeter := by
  sorry

#eval figure1Perimeter
#eval figure2Perimeter

end perimeter_difference_zero_l1368_136852


namespace boat_problem_l1368_136897

theorem boat_problem (boat1 boat2 boat3 boat4 boat5 : ℕ) 
  (h1 : boat1 = 2)
  (h2 : boat2 = 4)
  (h3 : boat3 = 3)
  (h4 : boat4 = 5)
  (h5 : boat5 = 6) :
  boat5 - (boat1 + boat2 + boat3 + boat4 + boat5) / 5 = 2 := by
  sorry

end boat_problem_l1368_136897


namespace product_digits_sum_l1368_136870

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digits_sum :
  sumOfDigitsBase7 (multiplyBase7 24 30) = 6 := by sorry

end product_digits_sum_l1368_136870


namespace reciprocal_sum_equals_two_l1368_136841

theorem reciprocal_sum_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 2) (h_prod : x * y = 1) : 
  1 / x + 1 / y = 2 := by
sorry

end reciprocal_sum_equals_two_l1368_136841


namespace power_difference_equals_one_l1368_136854

theorem power_difference_equals_one (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) :
  a^(m - 3*n) = 1 := by
  sorry

end power_difference_equals_one_l1368_136854


namespace smallest_b_for_g_nested_equals_g_l1368_136861

def g (x : ℤ) : ℤ :=
  if x % 15 = 0 then x / 15
  else if x % 3 = 0 then 5 * x
  else if x % 5 = 0 then 3 * x
  else x + 5

def g_nested (b : ℕ) (x : ℤ) : ℤ :=
  match b with
  | 0 => x
  | n + 1 => g (g_nested n x)

theorem smallest_b_for_g_nested_equals_g :
  ∀ b : ℕ, b > 1 → g_nested b 2 = g 2 → b ≥ 15 :=
sorry

end smallest_b_for_g_nested_equals_g_l1368_136861


namespace intersection_distance_sum_l1368_136894

/-- Given two lines in the Cartesian plane that intersect at point M,
    prove that the sum of squared distances from M to the fixed points P and Q is 10. -/
theorem intersection_distance_sum (a : ℝ) (M : ℝ × ℝ) :
  let P : ℝ × ℝ := (0, 1)
  let Q : ℝ × ℝ := (-3, 0)
  let l := {(x, y) : ℝ × ℝ | a * x + y - 1 = 0}
  let m := {(x, y) : ℝ × ℝ | x - a * y + 3 = 0}
  M ∈ l ∧ M ∈ m →
  (M.1 - P.1)^2 + (M.2 - P.2)^2 + (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 10 :=
by sorry

end intersection_distance_sum_l1368_136894


namespace perpendicular_vectors_l1368_136896

def a : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i • a i = 0) → m = 7 := by
  sorry

end perpendicular_vectors_l1368_136896


namespace garrison_provisions_theorem_l1368_136892

/-- Represents the number of days provisions last for a garrison -/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  let totalProvisions := initialMen * (daysBeforeReinforcement + daysAfterReinforcement)
  let remainingProvisions := totalProvisions - initialMen * daysBeforeReinforcement
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  (totalProvisions / initialMen : ℕ)

theorem garrison_provisions_theorem (initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement : ℕ) :
  initialMen = 2000 →
  reinforcementMen = 1300 →
  daysBeforeReinforcement = 21 →
  daysAfterReinforcement = 20 →
  provisionDays initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement = 54 := by
  sorry

#eval provisionDays 2000 1300 21 20

end garrison_provisions_theorem_l1368_136892


namespace inequality_proof_l1368_136832

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end inequality_proof_l1368_136832


namespace difference_of_squares_l1368_136814

theorem difference_of_squares : 303^2 - 297^2 = 3600 := by sorry

end difference_of_squares_l1368_136814


namespace problem_statement_l1368_136808

theorem problem_statement (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end problem_statement_l1368_136808


namespace jack_plates_problem_l1368_136829

theorem jack_plates_problem (flower_initial : ℕ) (checked : ℕ) (total_final : ℕ) :
  flower_initial = 4 →
  total_final = 27 →
  total_final = (flower_initial - 1) + checked + 2 * checked →
  checked = 8 := by
  sorry

end jack_plates_problem_l1368_136829


namespace herrings_caught_l1368_136807

/-- Given the total number of fish caught and the number of pikes and sturgeons,
    calculate the number of herrings caught. -/
theorem herrings_caught (total : ℕ) (pikes : ℕ) (sturgeons : ℕ) 
  (h1 : total = 145) (h2 : pikes = 30) (h3 : sturgeons = 40) :
  total - pikes - sturgeons = 75 := by
  sorry

end herrings_caught_l1368_136807


namespace sampled_bag_number_61st_group_l1368_136898

/-- Given a total number of bags, sample size, first sampled bag number, and group number,
    calculate the bag number for that group. -/
def sampledBagNumber (totalBags : ℕ) (sampleSize : ℕ) (firstSampledBag : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSampledBag + (groupNumber - 1) * (totalBags / sampleSize)

/-- Theorem stating that for the given conditions, the 61st group's sampled bag number is 1211. -/
theorem sampled_bag_number_61st_group :
  sampledBagNumber 3000 150 11 61 = 1211 := by
  sorry


end sampled_bag_number_61st_group_l1368_136898


namespace prob_unit_apart_value_l1368_136855

/-- A rectangle with 10 points spaced at unit intervals on corners and edge midpoints -/
structure UnitRectangle :=
  (width : ℕ)
  (height : ℕ)
  (total_points : ℕ)
  (h_width : width = 5)
  (h_height : height = 2)
  (h_total_points : total_points = 10)

/-- The number of pairs of points that are exactly one unit apart -/
def unit_apart_pairs (r : UnitRectangle) : ℕ := 13

/-- The total number of ways to choose two points from the rectangle -/
def total_pairs (r : UnitRectangle) : ℕ := r.total_points.choose 2

/-- The probability of selecting two points that are exactly one unit apart -/
def prob_unit_apart (r : UnitRectangle) : ℚ :=
  (unit_apart_pairs r : ℚ) / (total_pairs r : ℚ)

theorem prob_unit_apart_value (r : UnitRectangle) :
  prob_unit_apart r = 13 / 45 := by sorry

end prob_unit_apart_value_l1368_136855


namespace intersection_A_B_l1368_136871

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end intersection_A_B_l1368_136871


namespace opposite_of_negative_three_l1368_136848

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end opposite_of_negative_three_l1368_136848


namespace work_ratio_is_three_to_eight_l1368_136811

/-- Represents a worker's weekly schedule and earnings -/
structure WorkerSchedule where
  days_per_week : ℕ
  weekly_earnings : ℚ
  daily_salary : ℚ

/-- Calculates the ratio of time worked on the last three days to the first four days -/
def work_ratio (w : WorkerSchedule) : ℚ × ℚ :=
  let last_three_days_earnings := w.weekly_earnings - (4 * w.daily_salary)
  let last_three_days_time := last_three_days_earnings / w.daily_salary
  let first_four_days_time := 4
  (last_three_days_time * 2, first_four_days_time * 2)

/-- Theorem stating the work ratio for a specific worker schedule -/
theorem work_ratio_is_three_to_eight (w : WorkerSchedule) 
  (h1 : w.days_per_week = 7)
  (h2 : w.weekly_earnings = 55)
  (h3 : w.daily_salary = 10) :
  work_ratio w = (3, 8) := by
  sorry

#eval work_ratio ⟨7, 55, 10⟩

end work_ratio_is_three_to_eight_l1368_136811


namespace line_through_point_l1368_136825

theorem line_through_point (k : ℚ) :
  (1 - 3 * k * (1/2) = 10 * 3) ↔ (k = -58/3) := by sorry

end line_through_point_l1368_136825
