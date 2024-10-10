import Mathlib

namespace isosceles_triangle_vertex_angle_l1726_172629

/-- An isosceles triangle with two interior angles summing to 100° has a vertex angle of either 20° or 80°. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = b ∨ b = c ∨ a = c) →  -- Triangle is isosceles
  ((a + b = 100 ∧ c ≠ b) ∨ (b + c = 100 ∧ a ≠ b) ∨ (a + c = 100 ∧ a ≠ b)) →  -- Two angles sum to 100°
  (c = 20 ∨ c = 80) ∨ (a = 20 ∨ a = 80) ∨ (b = 20 ∨ b = 80) :=  -- Vertex angle is 20° or 80°
by sorry

end isosceles_triangle_vertex_angle_l1726_172629


namespace power_of_power_l1726_172637

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l1726_172637


namespace mod_seven_power_difference_l1726_172663

theorem mod_seven_power_difference : 47^2023 - 28^2023 ≡ 5 [ZMOD 7] := by
  sorry

end mod_seven_power_difference_l1726_172663


namespace distance_calculation_l1726_172623

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 84

theorem distance_calculation (distance : ℝ) : 
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time) → 
  distance = 210 := by
sorry

end distance_calculation_l1726_172623


namespace special_function_period_l1726_172687

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = (1 + f x) / (1 - f x)

/-- The period of a real function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The main theorem: if f is a SpecialFunction with parameter a, then it has period 4|a| -/
theorem special_function_period (f : ℝ → ℝ) (a : ℝ) 
    (hf : SpecialFunction f a) : HasPeriod f (4 * |a|) := by
  sorry

end special_function_period_l1726_172687


namespace lcm_problem_l1726_172653

theorem lcm_problem (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 := by
  sorry

end lcm_problem_l1726_172653


namespace cups_per_girl_l1726_172635

theorem cups_per_girl (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  2 * boys = total_students - boys →
  (total_students - boys) * (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 :=
by sorry

end cups_per_girl_l1726_172635


namespace chord_line_equation_l1726_172643

/-- The equation of a line passing through a chord of an ellipse, given the chord's midpoint -/
theorem chord_line_equation (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  M = (4, 2) →
  M.1 = (A.1 + B.1) / 2 →
  M.2 = (A.2 + B.2) / 2 →
  A.1^2 + 4 * A.2^2 = 36 →
  B.1^2 + 4 * B.2^2 = 36 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → x + 2*y - 8 = 0 :=
by sorry

end chord_line_equation_l1726_172643


namespace postal_stamps_theorem_l1726_172654

/-- The number of color stamps sold -/
def color_stamps : ℕ := 578833

/-- The total number of stamps sold -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold -/
def bw_stamps : ℕ := total_stamps - color_stamps

theorem postal_stamps_theorem : 
  bw_stamps = 523776 := by sorry

end postal_stamps_theorem_l1726_172654


namespace max_additional_pens_is_four_l1726_172674

def initial_amount : ℕ := 100
def remaining_amount : ℕ := 61
def pens_bought : ℕ := 3

def cost_per_pen : ℕ := (initial_amount - remaining_amount) / pens_bought

def max_additional_pens : ℕ := remaining_amount / cost_per_pen

theorem max_additional_pens_is_four :
  max_additional_pens = 4 := by sorry

end max_additional_pens_is_four_l1726_172674


namespace election_votes_calculation_l1726_172688

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (losing_candidate_votes winning_candidate_votes : ℕ),
    losing_candidate_votes = (32 * total_votes) / 100 ∧
    winning_candidate_votes = losing_candidate_votes + 1908 ∧
    winning_candidate_votes + losing_candidate_votes = total_votes) →
  total_votes = 5300 := by
sorry

end election_votes_calculation_l1726_172688


namespace seventh_term_is_twenty_l1726_172641

/-- An arithmetic sequence with first term 2 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Theorem stating that the 7th term of the arithmetic sequence is 20 -/
theorem seventh_term_is_twenty : arithmeticSequence 7 = 20 := by
  sorry

end seventh_term_is_twenty_l1726_172641


namespace tiles_count_l1726_172627

/-- Represents a square floor tiled with white tiles on the perimeter and black tiles in the center -/
structure TiledSquare where
  side_length : ℕ
  white_tiles : ℕ
  black_tiles : ℕ

/-- The number of white tiles on the perimeter of a square floor -/
def perimeter_tiles (s : TiledSquare) : ℕ := 4 * (s.side_length - 1)

/-- The number of black tiles in the center of a square floor -/
def center_tiles (s : TiledSquare) : ℕ := (s.side_length - 2)^2

/-- Theorem stating that if there are 80 white tiles on the perimeter, there are 361 black tiles in the center -/
theorem tiles_count (s : TiledSquare) :
  perimeter_tiles s = 80 → center_tiles s = 361 := by
  sorry

end tiles_count_l1726_172627


namespace correct_average_l1726_172616

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 55 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 19 := by
  sorry

#check correct_average

end correct_average_l1726_172616


namespace geometric_sequence_fifth_term_l1726_172697

/-- A geometric sequence with first term 1 and common ratio q ≠ -1 -/
def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  q^(n-1)

theorem geometric_sequence_fifth_term
  (q : ℝ)
  (h1 : q ≠ -1)
  (h2 : geometric_sequence q 5 + geometric_sequence q 4 = 3 * (geometric_sequence q 3 + geometric_sequence q 2)) :
  geometric_sequence q 5 = 9 :=
sorry

end geometric_sequence_fifth_term_l1726_172697


namespace sum_of_two_numbers_l1726_172655

theorem sum_of_two_numbers (a b : ℤ) : 
  (a = 2 * b - 43) → (min a b = 19) → (a + b = 14) := by
  sorry

end sum_of_two_numbers_l1726_172655


namespace count_perfect_square_factors_of_14400_l1726_172665

/-- The number of perfect square factors of 14400 -/
def perfect_square_factors_of_14400 : ℕ :=
  let n := 14400
  let prime_factorization := (2, 4) :: (3, 2) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of perfect square factors of 14400 is 12 -/
theorem count_perfect_square_factors_of_14400 :
  perfect_square_factors_of_14400 = 12 := by
  sorry

end count_perfect_square_factors_of_14400_l1726_172665


namespace fourth_student_id_l1726_172636

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_id : ℕ
  step : ℕ

/-- Creates a systematic sample given the total number of students and sample size -/
def create_systematic_sample (total : ℕ) (size : ℕ) : SystematicSample :=
  { total_students := total
  , sample_size := size
  , first_id := 3  -- Given in the problem
  , step := (total - 2) / size }

/-- Checks if a given ID is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = sample.first_id + k * sample.step ∧ k < sample.sample_size

/-- Main theorem: If 3, 29, and 42 are in the sample, then 16 is also in the sample -/
theorem fourth_student_id (sample : SystematicSample)
    (h_total : sample.total_students = 54)
    (h_size : sample.sample_size = 4)
    (h_3 : is_in_sample sample 3)
    (h_29 : is_in_sample sample 29)
    (h_42 : is_in_sample sample 42) :
    is_in_sample sample 16 := by
  sorry

end fourth_student_id_l1726_172636


namespace cos_BAD_equals_sqrt_13_45_l1726_172682

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the lengths of the sides
def AB (A B : ℝ × ℝ) : ℝ := sorry
def AC (A C : ℝ × ℝ) : ℝ := sorry
def BC (B C : ℝ × ℝ) : ℝ := sorry

-- Define a point D on BC
def D_on_BC (B C D : ℝ × ℝ) : Prop := sorry

-- Define the angle bisector property
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the cosine of an angle
def cos_angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem cos_BAD_equals_sqrt_13_45 
  (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : AB A B = 5)
  (h3 : AC A C = 9)
  (h4 : BC B C = 12)
  (h5 : D_on_BC B C D)
  (h6 : is_angle_bisector A B C D) :
  cos_angle B A D = Real.sqrt (13 / 45) := by
  sorry

end cos_BAD_equals_sqrt_13_45_l1726_172682


namespace rhombus_perimeter_l1726_172607

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 16 * Real.sqrt 13 := by
  sorry

end rhombus_perimeter_l1726_172607


namespace smaller_cube_volume_l1726_172648

theorem smaller_cube_volume 
  (large_cube_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (surface_area_diff : ℝ) : ℝ :=
by
  have h1 : large_cube_volume = 343 := by sorry
  have h2 : num_small_cubes = 343 := by sorry
  have h3 : surface_area_diff = 1764 := by sorry
  
  -- Define the side length of the large cube
  let large_side : ℝ := large_cube_volume ^ (1/3)
  
  -- Define the surface area of the large cube
  let large_surface_area : ℝ := 6 * large_side^2
  
  -- Define the volume of each small cube
  let small_cube_volume : ℝ := large_cube_volume / num_small_cubes
  
  -- Define the side length of each small cube
  let small_side : ℝ := small_cube_volume ^ (1/3)
  
  -- Define the total surface area of all small cubes
  let total_small_surface_area : ℝ := 6 * small_side^2 * num_small_cubes
  
  -- The main theorem
  have : small_cube_volume = 1 := by sorry

  exact small_cube_volume

end smaller_cube_volume_l1726_172648


namespace sock_pair_count_l1726_172664

/-- Given 8 pairs of socks, calculates the number of different pairs that can be formed
    by selecting 2 socks that are not from the same original pair -/
def sockPairs (totalPairs : Nat) : Nat :=
  let totalSocks := 2 * totalPairs
  let firstChoice := totalSocks
  let secondChoice := totalSocks - 2
  (firstChoice * secondChoice) / 2

/-- Theorem stating that with 8 pairs of socks, the number of different pairs
    that can be formed by selecting 2 socks not from the same original pair is 112 -/
theorem sock_pair_count : sockPairs 8 = 112 := by
  sorry

end sock_pair_count_l1726_172664


namespace min_type_b_workers_l1726_172640

/-- The number of workers in the workshop -/
def total_workers : ℕ := 20

/-- The number of Type A parts a worker can produce per day -/
def type_a_production : ℕ := 6

/-- The number of Type B parts a worker can produce per day -/
def type_b_production : ℕ := 5

/-- The profit (in yuan) from producing one Type A part -/
def type_a_profit : ℕ := 150

/-- The profit (in yuan) from producing one Type B part -/
def type_b_profit : ℕ := 260

/-- The daily profit function (in yuan) based on the number of workers producing Type A parts -/
def daily_profit (x : ℝ) : ℝ :=
  type_a_profit * type_a_production * x + type_b_profit * type_b_production * (total_workers - x)

/-- The minimum required daily profit (in yuan) -/
def min_profit : ℝ := 24000

theorem min_type_b_workers :
  ∀ x : ℝ, 0 ≤ x → x ≤ total_workers →
  (∀ y : ℝ, y ≥ min_profit → daily_profit x ≥ y) →
  total_workers - x ≥ 15 :=
by sorry

end min_type_b_workers_l1726_172640


namespace fraction_ratio_equality_l1726_172638

theorem fraction_ratio_equality : ∃ x : ℚ, (3 / 7) / (6 / 5) = x / (2 / 5) ∧ x = 1 / 7 := by
  sorry

end fraction_ratio_equality_l1726_172638


namespace divisibility_condition_l1726_172602

theorem divisibility_condition (p : Nat) (α : Nat) (x : Int) :
  Prime p → p > 2 → α > 0 →
  (∃ k : Int, x^2 - 1 = k * p^α) ↔
  (∃ t : Int, x = t * p^α + 1 ∨ x = t * p^α - 1) := by
  sorry

end divisibility_condition_l1726_172602


namespace total_distance_covered_l1726_172642

-- Define the given conditions
def cycling_time : ℚ := 30 / 60  -- 30 minutes in hours
def cycling_rate : ℚ := 12       -- 12 mph
def skating_time : ℚ := 45 / 60  -- 45 minutes in hours
def skating_rate : ℚ := 8        -- 8 mph
def total_time : ℚ := 75 / 60    -- 1 hour and 15 minutes in hours

-- State the theorem
theorem total_distance_covered : 
  cycling_time * cycling_rate + skating_time * skating_rate = 12 := by
  sorry -- The proof is omitted as per instructions

end total_distance_covered_l1726_172642


namespace negation_of_universal_proposition_l1726_172631

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (x₀ + 1) ≤ 0) :=
by sorry

end negation_of_universal_proposition_l1726_172631


namespace original_books_l1726_172617

/-- The number of books person A has -/
def books_A : ℕ := sorry

/-- The number of books person B has -/
def books_B : ℕ := sorry

/-- If A gives 10 books to B, they have an equal number of books -/
axiom equal_books : books_A - 10 = books_B + 10

/-- If B gives 10 books to A, A has twice the number of books B has left -/
axiom double_books : books_A + 10 = 2 * (books_B - 10)

theorem original_books : books_A = 70 ∧ books_B = 50 := by sorry

end original_books_l1726_172617


namespace train_length_calculation_l1726_172666

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ (length_m : ℝ), abs (length_m - 200.04) < 0.01 ∧ length_m = (speed_kmh * 1000 / 3600) * time_s :=
by sorry

end train_length_calculation_l1726_172666


namespace sector_area_l1726_172612

/-- The area of a circular sector with radius 6 cm and central angle 120° is 12π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let θ : ℝ := 120
  let π : ℝ := Real.pi
  (θ / 360) * π * r^2 = 12 * π := by sorry

end sector_area_l1726_172612


namespace barrel_contents_l1726_172673

theorem barrel_contents :
  ∀ (x : ℝ),
  (x > 0) →
  (x / 6 = x - 5 * x / 6) →
  (5 * x / 30 = 5 * x / 6 - 2 * x / 3) →
  (x / 6 = 2 * x / 3 - x / 2) →
  ((x + 120) + (5 * x / 6 + 120) = 4 * (x / 2)) →
  (x = 1440 ∧ 
   5 * x / 6 = 1200 ∧ 
   2 * x / 3 = 960 ∧ 
   x / 2 = 720) :=
by sorry

end barrel_contents_l1726_172673


namespace power_equation_solution_l1726_172610

theorem power_equation_solution (K : ℕ) : 32^4 * 4^6 = 2^K → K = 32 := by
  sorry

end power_equation_solution_l1726_172610


namespace long_jump_competition_l1726_172675

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third < second →
  fourth = third + 3 →
  fourth = 24 →
  second - third = 2 :=
by sorry

end long_jump_competition_l1726_172675


namespace age_height_not_function_l1726_172672

-- Define a type for age and height
def Age := ℕ
def Height := ℝ

-- Define a relation between age and height
def AgeHeightRelation := Age → Set Height

-- Define what it means for a relation to be a function
def IsFunction (R : α → Set β) : Prop :=
  ∀ x : α, ∃! y : β, y ∈ R x

-- State the theorem
theorem age_height_not_function :
  ∃ R : AgeHeightRelation, ¬ IsFunction R :=
sorry

end age_height_not_function_l1726_172672


namespace sum_of_squares_of_roots_l1726_172683

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 7 * p^2 + 2 * p - 4 = 0) →
  (3 * q^3 - 7 * q^2 + 2 * q - 4 = 0) →
  (3 * r^3 - 7 * r^2 + 2 * r - 4 = 0) →
  p^2 + q^2 + r^2 = 37/9 := by
sorry

end sum_of_squares_of_roots_l1726_172683


namespace smallest_prime_with_condition_l1726_172625

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_prime_with_condition : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    is_two_digit p ∧ 
    tens_digit p = 3 ∧ 
    ¬(Prime (reverse_digits p + 5)) ∧
    ∀ (q : ℕ), Prime q ∧ is_two_digit q ∧ tens_digit q = 3 ∧ ¬(Prime (reverse_digits q + 5)) → p ≤ q ∧
    p = 31 :=
by sorry

end smallest_prime_with_condition_l1726_172625


namespace only_cone_cannot_have_rectangular_cross_section_l1726_172662

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.Cone => False
  | GeometricSolid.PentagonalPrism => True
  | GeometricSolid.Cube => True

-- Theorem stating that only the cone cannot have a rectangular cross-section
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end only_cone_cannot_have_rectangular_cross_section_l1726_172662


namespace not_right_triangle_with_angle_ratio_l1726_172618

theorem not_right_triangle_with_angle_ratio (A B C : ℝ) (h : A + B + C = 180) 
  (ratio : A / 3 = B / 4 ∧ A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end not_right_triangle_with_angle_ratio_l1726_172618


namespace total_tiles_from_black_tiles_total_tiles_is_2601_l1726_172680

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  size : ℕ
  blackTilesCount : ℕ

/-- Theorem stating the relationship between the number of black tiles and total tiles -/
theorem total_tiles_from_black_tiles (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) : 
  floor.size * floor.size = 2601 := by
  sorry

/-- Main theorem proving the total number of tiles -/
theorem total_tiles_is_2601 (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) 
  (h2 : floor.blackTilesCount = 2 * floor.size - 1) : 
  floor.size * floor.size = 2601 := by
  sorry

end total_tiles_from_black_tiles_total_tiles_is_2601_l1726_172680


namespace max_value_P_l1726_172668

open Real

/-- Given positive real numbers a, b, and c satisfying abc + a + c = b,
    the maximum value of P = 2/(a² + 1) - 2/(b² + 1) + 3/(c² + 1) is 1. -/
theorem max_value_P (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a * b * c + a + c = b) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → x * y * z + x + z = y →
    2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ≤ M :=
by sorry

end max_value_P_l1726_172668


namespace legos_set_cost_l1726_172691

theorem legos_set_cost (total_earnings : ℕ) (num_cars : ℕ) (car_price : ℕ) (legos_price : ℕ) :
  total_earnings = 45 →
  num_cars = 3 →
  car_price = 5 →
  total_earnings = num_cars * car_price + legos_price →
  legos_price = 30 := by
sorry

end legos_set_cost_l1726_172691


namespace cubic_expression_evaluation_l1726_172689

theorem cubic_expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end cubic_expression_evaluation_l1726_172689


namespace hostel_mess_expenditure_l1726_172696

/-- The original expenditure of a hostel mess given specific conditions -/
theorem hostel_mess_expenditure :
  ∀ (initial_students : ℕ) 
    (student_increase : ℕ) 
    (expense_increase : ℕ) 
    (avg_expenditure_decrease : ℕ),
  initial_students = 35 →
  student_increase = 7 →
  expense_increase = 42 →
  avg_expenditure_decrease = 1 →
  ∃ (original_expenditure : ℕ),
    original_expenditure = 420 ∧
    (initial_students + student_increase) * 
      ((original_expenditure / initial_students) - avg_expenditure_decrease) =
    original_expenditure + expense_increase :=
by sorry

end hostel_mess_expenditure_l1726_172696


namespace income_remainder_relation_l1726_172694

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children : ℝ
  wife : ℝ
  bills : ℝ
  savings : ℝ
  remainder : ℝ

/-- Theorem stating the relationship between income and remainder --/
theorem income_remainder_relation (d : IncomeDistribution) :
  d.children = 0.18 * d.total ∧
  d.wife = 0.28 * d.total ∧
  d.bills = 0.12 * d.total ∧
  d.savings = 0.15 * d.total ∧
  d.remainder = 35000 →
  0.27 * d.total = 35000 := by
  sorry

end income_remainder_relation_l1726_172694


namespace auction_price_increase_l1726_172676

/-- Represents an auction with a starting price, ending price, number of bidders, and bids per bidder -/
structure Auction where
  start_price : ℕ
  end_price : ℕ
  num_bidders : ℕ
  bids_per_bidder : ℕ

/-- Calculates the price increase per bid in an auction -/
def price_increase_per_bid (a : Auction) : ℚ :=
  (a.end_price - a.start_price : ℚ) / (a.num_bidders * a.bids_per_bidder : ℚ)

/-- Theorem stating that for the given auction conditions, the price increase per bid is $5 -/
theorem auction_price_increase (a : Auction)
  (h1 : a.start_price = 15)
  (h2 : a.end_price = 65)
  (h3 : a.num_bidders = 2)
  (h4 : a.bids_per_bidder = 5) :
  price_increase_per_bid a = 5 := by
  sorry

end auction_price_increase_l1726_172676


namespace power_sum_and_quotient_l1726_172632

theorem power_sum_and_quotient : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end power_sum_and_quotient_l1726_172632


namespace lees_initial_money_l1726_172657

def friends_money : ℕ := 8
def meal_cost : ℕ := 15
def total_paid : ℕ := 18

theorem lees_initial_money :
  ∃ (lees_money : ℕ), lees_money + friends_money = total_paid ∧ lees_money = 10 := by
sorry

end lees_initial_money_l1726_172657


namespace parabola_min_distance_l1726_172677

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum distance from any point on the parabola to its focus is 1 -/
def min_distance_to_focus (para : Parabola) : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * para.p * x ∧ 1 ≤ Real.sqrt ((x - para.p/2)^2 + y^2)

/-- If the minimum distance from any point on the parabola to the focus is 1, then p = 2 -/
theorem parabola_min_distance (para : Parabola) 
    (h_min : min_distance_to_focus para) : para.p = 2 := by
  sorry

end parabola_min_distance_l1726_172677


namespace root_sum_of_coefficients_l1726_172606

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : Prop :=
  x^2 + p * x + q = 0

-- State the theorem
theorem root_sum_of_coefficients (p q : ℝ) :
  quadratic p q (1 + i) → p + q = 0 := by
  sorry

end root_sum_of_coefficients_l1726_172606


namespace a_b_reciprocals_l1726_172685

theorem a_b_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end a_b_reciprocals_l1726_172685


namespace intersection_A_complement_B_l1726_172690

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end intersection_A_complement_B_l1726_172690


namespace polynomial_factorization_l1726_172681

theorem polynomial_factorization (a b : ℝ) (x : ℝ) :
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x) * (1 + x^2) * (a + b*x + b*x^2 + b*x^3) := by
  sorry

end polynomial_factorization_l1726_172681


namespace greatest_integer_satisfying_conditions_l1726_172667

theorem greatest_integer_satisfying_conditions : ∃ n : ℕ, 
  n < 200 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧
  ∃ l : ℕ, n + 4 = 10 * l ∧
  ∀ m : ℕ, m < 200 → 
    (∃ p : ℕ, m + 2 = 9 * p) → 
    (∃ q : ℕ, m + 4 = 10 * q) → 
    m ≤ n ∧
  n = 166 :=
by sorry

end greatest_integer_satisfying_conditions_l1726_172667


namespace jerry_piercing_pricing_l1726_172684

theorem jerry_piercing_pricing (nose_price : ℝ) (total_revenue : ℝ) (num_noses : ℕ) (num_ears : ℕ) :
  nose_price = 20 →
  total_revenue = 390 →
  num_noses = 6 →
  num_ears = 9 →
  let ear_price := (total_revenue - nose_price * num_noses) / num_ears
  let percentage_increase := (ear_price - nose_price) / nose_price * 100
  percentage_increase = 50 := by
sorry


end jerry_piercing_pricing_l1726_172684


namespace sequence_properties_l1726_172624

theorem sequence_properties (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) :
  (∀ n : ℕ+, S n = -n.val^2 + 24*n.val) →
  (∀ n : ℕ+, a n = S n - S (n-1)) →
  (∀ n : ℕ+, a n = -2*n.val + 25) ∧
  (∀ n : ℕ+, S n ≤ S 12) ∧
  (S 12 = 144) := by
sorry

end sequence_properties_l1726_172624


namespace lucca_basketball_percentage_proof_l1726_172656

/-- The percentage of Lucca's balls that are basketballs -/
def lucca_basketball_percentage : ℝ := 10

theorem lucca_basketball_percentage_proof :
  let lucca_total_balls : ℕ := 100
  let lucien_total_balls : ℕ := 200
  let lucien_basketball_percentage : ℝ := 20
  let total_basketballs : ℕ := 50
  lucca_basketball_percentage = 10 := by
  sorry

end lucca_basketball_percentage_proof_l1726_172656


namespace euler_family_mean_age_l1726_172615

def euler_family_ages : List ℝ := [5, 5, 10, 15, 8, 12, 16]

theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  sum_ages / num_children = 10.14 := by
sorry

end euler_family_mean_age_l1726_172615


namespace squirrel_acorns_l1726_172698

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (additional_acorns : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  additional_acorns = 15 →
  ∃ (acorns_per_squirrel : ℕ),
    acorns_per_squirrel = 130 ∧
    num_squirrels * (acorns_per_squirrel - additional_acorns) = total_acorns :=
by sorry

end squirrel_acorns_l1726_172698


namespace sin_cos_sum_equals_sqrt3_over_2_l1726_172669

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (43 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (43 * π / 180) * Real.sin (17 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt3_over_2_l1726_172669


namespace sock_pair_combinations_l1726_172634

def num_white_socks : ℕ := 4
def num_brown_socks : ℕ := 4
def num_blue_socks : ℕ := 4
def num_red_socks : ℕ := 4

def total_socks : ℕ := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

theorem sock_pair_combinations :
  (num_red_socks * num_white_socks) + 
  (num_red_socks * num_brown_socks) + 
  (num_red_socks * num_blue_socks) = 48 :=
by sorry

end sock_pair_combinations_l1726_172634


namespace rene_received_300_l1726_172670

-- Define the amounts given to each person
def rene_amount : ℝ := sorry
def florence_amount : ℝ := sorry
def isha_amount : ℝ := sorry

-- Define the theorem
theorem rene_received_300 
  (h1 : florence_amount = 3 * rene_amount)
  (h2 : isha_amount = florence_amount / 2)
  (h3 : rene_amount + florence_amount + isha_amount = 1650)
  : rene_amount = 300 := by
  sorry

end rene_received_300_l1726_172670


namespace carmen_jethro_ratio_l1726_172644

-- Define the amounts of money for each person
def jethro_money : ℚ := 20
def patricia_money : ℚ := 60
def carmen_money : ℚ := 113 - jethro_money - patricia_money

-- Define the conditions
axiom patricia_triple_jethro : patricia_money = 3 * jethro_money
axiom total_money : carmen_money + jethro_money + patricia_money = 113
axiom carmen_multiple_after : ∃ (m : ℚ), carmen_money + 7 = m * jethro_money

-- Theorem to prove
theorem carmen_jethro_ratio :
  (carmen_money + 7) / jethro_money = 2 := by sorry

end carmen_jethro_ratio_l1726_172644


namespace lawn_care_supplies_l1726_172699

theorem lawn_care_supplies (blade_cost : ℕ) (string_cost : ℕ) (total_cost : ℕ) (num_blades : ℕ) :
  blade_cost = 8 →
  string_cost = 7 →
  total_cost = 39 →
  blade_cost * num_blades + string_cost = total_cost →
  num_blades = 4 := by
sorry

end lawn_care_supplies_l1726_172699


namespace hotel_loss_calculation_l1726_172639

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
sorry

end hotel_loss_calculation_l1726_172639


namespace wade_friday_customers_l1726_172633

/-- Represents the number of customers Wade served on Friday -/
def F : ℕ := by sorry

/-- Wade's tip per customer in dollars -/
def tip_per_customer : ℚ := 2

/-- Total tips Wade made over the three days in dollars -/
def total_tips : ℚ := 296

theorem wade_friday_customers :
  F = 28 ∧
  tip_per_customer * (F + 3 * F + 36) = total_tips :=
by sorry

end wade_friday_customers_l1726_172633


namespace brocard_point_characterization_l1726_172603

open Real

/-- Triangle structure with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Point structure with coordinates x, y -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given side lengths -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a triangle given three points -/
def areaFromPoints (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Definition of Brocard point -/
def isBrocardPoint (p : Point) (t : Triangle) : Prop :=
  let s_abc := triangleArea t
  let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
  let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
  let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
  isInside p t ∧
  (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
  (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
  (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))

/-- Theorem: Characterization of Brocard point -/
theorem brocard_point_characterization (t : Triangle) (p : Point) :
  isBrocardPoint p t ↔
  (let s_abc := triangleArea t
   let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
   let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
   let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
   isInside p t ∧
   (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
   (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
   (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))) :=
by sorry

end brocard_point_characterization_l1726_172603


namespace parallel_vectors_imply_m_value_l1726_172649

/-- 
Given a non-zero vector a = (m^2 - 1, m + 1) that is parallel to vector b = (1, -2),
prove that m = 1/2.
-/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Condition 1: Vector a is non-zero
  ∃ (k : ℝ), k ≠ 0 ∧ k * (m^2 - 1) = 1 ∧ k * (m + 1) = -2 →  -- Condition 2 and 3: Parallel vectors
  m = 1/2 := by
sorry

end parallel_vectors_imply_m_value_l1726_172649


namespace apple_picking_ratio_l1726_172678

theorem apple_picking_ratio :
  ∀ (first_hour second_hour third_hour : ℕ),
    first_hour = 66 →
    second_hour = 2 * first_hour →
    first_hour + second_hour + third_hour = 220 →
    third_hour * 3 = first_hour :=
by
  sorry

end apple_picking_ratio_l1726_172678


namespace modular_inverse_of_5_mod_221_l1726_172600

theorem modular_inverse_of_5_mod_221 : ∃ x : ℕ, x < 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end modular_inverse_of_5_mod_221_l1726_172600


namespace equilateral_triangle_perimeter_l1726_172604

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ

-- Define the given conditions
def given_triangle (x : ℝ) : Prop :=
  ∃ (t : EquilateralTriangle), t.side_length = 2*x ∧ t.side_length = x + 15

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_perimeter :
  ∀ x : ℝ, given_triangle x → ∃ (t : EquilateralTriangle), perimeter t = 90 :=
by sorry

end equilateral_triangle_perimeter_l1726_172604


namespace select_three_from_eight_l1726_172693

theorem select_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end select_three_from_eight_l1726_172693


namespace total_cost_is_correct_l1726_172605

/-- Represents a country --/
inductive Country
| Italy
| Germany

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties

/-- The price of a stamp in cents --/
def stampPrice (c : Country) : ℕ :=
  match c with
  | Country.Italy => 7
  | Country.Germany => 5

/-- The number of stamps Juan has from a given country and decade --/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 8
  | Country.Germany, Decade.Fifties => 7
  | Country.Germany, Decade.Sixties => 6

/-- The total cost of Juan's European stamps from Italy and Germany issued before the 70's --/
def totalCost : ℚ :=
  let italyTotal := (stampCount Country.Italy Decade.Fifties + stampCount Country.Italy Decade.Sixties) * stampPrice Country.Italy
  let germanyTotal := (stampCount Country.Germany Decade.Fifties + stampCount Country.Germany Decade.Sixties) * stampPrice Country.Germany
  (italyTotal + germanyTotal : ℚ) / 100

theorem total_cost_is_correct : totalCost = 156 / 100 := by
  sorry

end total_cost_is_correct_l1726_172605


namespace triangle_intersection_height_l1726_172628

theorem triangle_intersection_height (t : ℝ) : 
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (8, 0)
  let T : ℝ × ℝ := ((8 - t) / 4, t)
  let U : ℝ × ℝ := (8 - t, t)
  let area_ATU : ℝ := (1 / 2) * (U.1 - T.1) * (A.2 - T.2)
  (0 ≤ t) ∧ (t ≤ 8) ∧ (area_ATU = 13.5) → t = 2 :=
by sorry

end triangle_intersection_height_l1726_172628


namespace estimate_student_population_l1726_172659

theorem estimate_student_population (first_survey : ℕ) (second_survey : ℕ) (overlap : ℕ) 
  (h1 : first_survey = 80)
  (h2 : second_survey = 100)
  (h3 : overlap = 20) :
  (first_survey * second_survey) / overlap = 400 := by
  sorry

end estimate_student_population_l1726_172659


namespace game_outcome_theorem_l1726_172652

/-- Represents the outcome of the game for a player -/
inductive Outcome
  | Points (n : ℕ)
  | NoPoints

/-- Represents a player's choice in the game -/
structure PlayerChoice where
  value : ℕ
  is_valid : 0 ≤ value ∧ value ≤ 10

/-- Determines the outcome for a player based on their choice and whether it's unique -/
def gameOutcome (choice : PlayerChoice) (is_unique : Bool) : Outcome :=
  if is_unique then Outcome.Points choice.value else Outcome.NoPoints

/-- Theorem stating that the outcome is either the chosen points or zero -/
theorem game_outcome_theorem (choice : PlayerChoice) (is_unique : Bool) :
  (gameOutcome choice is_unique = Outcome.Points choice.value) ∨
  (gameOutcome choice is_unique = Outcome.NoPoints) := by
  sorry

#check game_outcome_theorem

end game_outcome_theorem_l1726_172652


namespace quadratic_two_roots_l1726_172626

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end quadratic_two_roots_l1726_172626


namespace series_sum_equals_one_six_hundredth_l1726_172671

/-- The sum of the series Σ(6n + 2) / ((6n - 1)^2 * (6n + 5)^2) from n=1 to infinity equals 1/600. -/
theorem series_sum_equals_one_six_hundredth :
  ∑' n : ℕ, (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end series_sum_equals_one_six_hundredth_l1726_172671


namespace find_coefficient_l1726_172621

/-- Given a polynomial equation and a sum condition, prove the value of a specific coefficient. -/
theorem find_coefficient (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - a) * (x + 2)^5 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -96) →
  a₄ = -10 :=
by sorry

end find_coefficient_l1726_172621


namespace two_correct_conclusions_l1726_172611

theorem two_correct_conclusions : ∃ (S : Finset (Prop)), S.card = 2 ∧ S ⊆ 
  {∀ (k b x₁ x₂ y₁ y₂ : ℝ), k < 0 → y₁ = k * x₁ + b → y₂ = k * x₂ + b → x₁ > x₂ → y₁ > y₂,
   ∀ (k b : ℝ), (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x + b) → 
                k > 0 ∧ b > 0,
   ∀ (m : ℝ), (m - 1) * 0 + m^2 + 2 = 3 → m = 1 ∨ m = -1} ∧ 
  (∀ p ∈ S, p) := by
sorry

end two_correct_conclusions_l1726_172611


namespace square_equation_solution_l1726_172692

theorem square_equation_solution :
  ∀ x : ℝ, x^2 = 16 ↔ x = -4 ∨ x = 4 := by sorry

end square_equation_solution_l1726_172692


namespace expression_equals_one_l1726_172650

theorem expression_equals_one : 
  (150^2 - 12^2) / (90^2 - 18^2) * ((90 - 18)*(90 + 18)) / ((150 - 12)*(150 + 12)) = 1 := by
  sorry

end expression_equals_one_l1726_172650


namespace parallelogram_division_max_parts_l1726_172613

/-- Given a parallelogram divided into a grid of M by N parts, with one additional line drawn,
    the maximum number of parts the parallelogram can be divided into is MN + M + N - 1. -/
theorem parallelogram_division_max_parts (M N : ℕ) :
  let initial_parts := M * N
  let additional_parts := M + N - 1
  initial_parts + additional_parts = M * N + M + N - 1 := by
  sorry

end parallelogram_division_max_parts_l1726_172613


namespace parentheses_multiplication_l1726_172620

theorem parentheses_multiplication : (4 - 3) * 2 = 2 := by
  sorry

end parentheses_multiplication_l1726_172620


namespace eggs_to_examine_l1726_172614

def number_of_trays : ℕ := 7
def eggs_per_tray : ℕ := 10
def percentage_to_examine : ℚ := 70 / 100

theorem eggs_to_examine :
  (number_of_trays * (eggs_per_tray * percentage_to_examine).floor) = 49 := by
  sorry

end eggs_to_examine_l1726_172614


namespace vanya_more_heads_probability_vanya_more_heads_probability_is_half_l1726_172679

/-- The probability that Vanya gets more heads than Tanya when Vanya flips a coin n+1 times and Tanya flips a coin n times. -/
theorem vanya_more_heads_probability (n : ℕ) : ℝ :=
  let vanya_flips := n + 1
  let tanya_flips := n
  let prob_vanya_more_heads := (1 : ℝ) / 2
  prob_vanya_more_heads

/-- Proof that the probability of Vanya getting more heads than Tanya is 1/2. -/
theorem vanya_more_heads_probability_is_half (n : ℕ) :
  vanya_more_heads_probability n = (1 : ℝ) / 2 := by
  sorry

end vanya_more_heads_probability_vanya_more_heads_probability_is_half_l1726_172679


namespace barrelCapacitiesSolution_l1726_172661

/-- Represents the capacities of three barrels --/
structure BarrelCapacities where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given capacities satisfy the problem conditions --/
def satisfiesConditions (c : BarrelCapacities) : Prop :=
  -- After first transfer, 1/4 remains in first barrel
  c.second = (3 * c.first) / 4 ∧
  -- After second transfer, 2/9 remains in second barrel
  c.third = (7 * c.first) / 12 ∧
  -- After third transfer, 50 more units needed to fill first barrel
  c.third + 50 = c.first

/-- The theorem to prove --/
theorem barrelCapacitiesSolution :
  ∃ (c : BarrelCapacities), satisfiesConditions c ∧ c.first = 120 ∧ c.second = 90 ∧ c.third = 70 := by
  sorry

end barrelCapacitiesSolution_l1726_172661


namespace sum_divisible_by_three_probability_l1726_172658

/-- Given a sequence of positive integers, the probability that the sum of three
    independently and randomly selected elements is divisible by 3 is at least 1/4. -/
theorem sum_divisible_by_three_probability (n : ℕ) (seq : Fin n → ℕ+) :
  ∃ (p q r : ℝ), p + q + r = 1 ∧ p ≥ 0 ∧ q ≥ 0 ∧ r ≥ 0 ∧
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by
  sorry

end sum_divisible_by_three_probability_l1726_172658


namespace special_rectangle_perimeter_l1726_172608

/-- A rectangle with integer dimensions where the area equals the perimeter minus 4 -/
structure SpecialRectangle where
  length : ℕ
  width : ℕ
  not_square : length ≠ width
  area_perimeter_relation : length * width = 2 * (length + width) - 4

/-- The perimeter of a SpecialRectangle is 26 -/
theorem special_rectangle_perimeter (r : SpecialRectangle) : 2 * (r.length + r.width) = 26 := by
  sorry

end special_rectangle_perimeter_l1726_172608


namespace greatest_negative_root_of_sine_cosine_equation_l1726_172647

theorem greatest_negative_root_of_sine_cosine_equation :
  let α : ℝ := Real.arctan (1 / 8)
  let β : ℝ := Real.arctan (4 / 7)
  let root : ℝ := (α + β - 2 * Real.pi) / 9
  (∀ x : ℝ, x < 0 → Real.sin x + 8 * Real.cos x = 4 * Real.sin (8 * x) + 7 * Real.cos (8 * x) → x ≤ root) ∧
  Real.sin root + 8 * Real.cos root = 4 * Real.sin (8 * root) + 7 * Real.cos (8 * root) ∧
  root < 0 :=
by sorry

end greatest_negative_root_of_sine_cosine_equation_l1726_172647


namespace repeating_decimal_reciprocal_l1726_172609

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem repeating_decimal_reciprocal :
  (repeating_decimal)⁻¹ = reciprocal := by sorry

end repeating_decimal_reciprocal_l1726_172609


namespace inequality_solution_l1726_172630

theorem inequality_solution (p q : ℝ) :
  q > 0 →
  (3 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 2 * p^2 * q ↔
  p ≥ 0 ∧ p < 3 :=
by sorry

end inequality_solution_l1726_172630


namespace building_height_l1726_172645

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height / flagpole_shadow) * building_shadow = 26 := by
  sorry

end building_height_l1726_172645


namespace square_difference_l1726_172619

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end square_difference_l1726_172619


namespace minimum_bundle_price_l1726_172651

/- Define the costs of items -/
def water_cost : ℚ := 0.50
def fruit_cost : ℚ := 0.25
def snack_cost : ℚ := 1.00

/- Define the bundle composition -/
def water_per_bundle : ℕ := 1
def snacks_per_bundle : ℕ := 3
def fruits_per_bundle : ℕ := 2

/- Define the special offer -/
def special_bundle_interval : ℕ := 5
def special_bundle_price : ℚ := 2.00
def complimentary_snacks : ℕ := 1

/- Theorem statement -/
theorem minimum_bundle_price (P : ℚ) : 
  (P ≥ 4.75) ↔ 
  (4 * P + special_bundle_price ≥ 
    5 * (water_cost * water_per_bundle + 
         snack_cost * snacks_per_bundle + 
         fruit_cost * fruits_per_bundle) + 
    snack_cost * complimentary_snacks) := by
  sorry

end minimum_bundle_price_l1726_172651


namespace square_perimeter_ratio_l1726_172622

theorem square_perimeter_ratio (d1 d11 s1 s11 P1 P11 : ℝ) : 
  d1 > 0 → 
  d11 = 11 * d1 → 
  d1 = s1 * Real.sqrt 2 → 
  d11 = s11 * Real.sqrt 2 → 
  P1 = 4 * s1 → 
  P11 = 4 * s11 → 
  P11 / P1 = 11 := by
sorry


end square_perimeter_ratio_l1726_172622


namespace egg_supply_solution_l1726_172601

/-- Represents the egg supply problem for Mark's farm --/
def egg_supply_problem (daily_supply_store1 : ℕ) (weekly_total : ℕ) : Prop :=
  ∃ (daily_supply_store2 : ℕ),
    daily_supply_store1 = 5 * 12 ∧
    weekly_total = 7 * (daily_supply_store1 + daily_supply_store2) ∧
    daily_supply_store2 = 30

/-- Theorem stating the solution to the egg supply problem --/
theorem egg_supply_solution : 
  egg_supply_problem 60 630 := by
  sorry

end egg_supply_solution_l1726_172601


namespace marble_197_is_red_l1726_172660

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 197th marble is red -/
theorem marble_197_is_red : marbleColor 197 = MarbleColor.Red :=
sorry

end marble_197_is_red_l1726_172660


namespace alberts_to_marys_age_ratio_l1726_172695

theorem alberts_to_marys_age_ratio (albert_age mary_age betty_age : ℕ) : 
  betty_age = 4 → 
  albert_age = 4 * betty_age → 
  mary_age = albert_age - 8 → 
  (albert_age : ℚ) / mary_age = 2 := by
sorry

end alberts_to_marys_age_ratio_l1726_172695


namespace redbirds_count_l1726_172686

theorem redbirds_count (total : ℕ) (bluebird_fraction : ℚ) (h1 : total = 120) (h2 : bluebird_fraction = 5/6) :
  (1 - bluebird_fraction) * total = 20 := by
  sorry

end redbirds_count_l1726_172686


namespace proper_subsets_count_l1726_172646

def U : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {1,2}
def B : Finset Nat := {3,4}

theorem proper_subsets_count :
  (Finset.powerset (A ∩ (U \ B))).card - 1 = 3 := by sorry

end proper_subsets_count_l1726_172646
