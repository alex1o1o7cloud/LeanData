import Mathlib

namespace NUMINAMATH_CALUDE_course_length_is_300_l4074_407421

/-- Represents the dogsled race scenario -/
structure DogsledRace where
  teamT_speed : ℝ
  teamA_speed_diff : ℝ
  teamT_time : ℝ
  teamA_time_diff : ℝ

/-- Calculates the length of the dogsled race course -/
def course_length (race : DogsledRace) : ℝ :=
  race.teamT_speed * race.teamT_time

/-- Theorem stating that the course length is 300 miles given the race conditions -/
theorem course_length_is_300 (race : DogsledRace)
  (h1 : race.teamT_speed = 20)
  (h2 : race.teamA_speed_diff = 5)
  (h3 : race.teamA_time_diff = 3)
  (h4 : race.teamT_time * race.teamT_speed = (race.teamT_time - race.teamA_time_diff) * (race.teamT_speed + race.teamA_speed_diff)) :
  course_length race = 300 := by
  sorry

#eval course_length { teamT_speed := 20, teamA_speed_diff := 5, teamT_time := 15, teamA_time_diff := 3 }

end NUMINAMATH_CALUDE_course_length_is_300_l4074_407421


namespace NUMINAMATH_CALUDE_distinct_sums_lower_bound_l4074_407450

theorem distinct_sums_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, 0 < a i) :
  (Finset.powerset (Finset.range n)).card ≥ n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_lower_bound_l4074_407450


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l4074_407435

theorem prime_quadratic_roots (p : ℕ) (h_prime : Nat.Prime p) 
  (h_roots : ∃ x y : ℤ, x ^ 2 - p * x - 580 * p = 0 ∧ y ^ 2 - p * y - 580 * p = 0 ∧ x ≠ y) :
  20 < p ∧ p < 30 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l4074_407435


namespace NUMINAMATH_CALUDE_linear_system_sum_fraction_l4074_407420

theorem linear_system_sum_fraction (a b c x y z : ℝ) 
  (eq1 : 5 * x + b * y + c * z = 0)
  (eq2 : a * x + 7 * y + c * z = 0)
  (eq3 : a * x + b * y + 9 * z = 0)
  (ha : a ≠ 5)
  (hx : x ≠ 0) :
  a / (a - 5) + b / (b - 7) + c / (c - 9) = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_system_sum_fraction_l4074_407420


namespace NUMINAMATH_CALUDE_five_graduates_three_companies_l4074_407455

/-- The number of ways to assign n graduates to k companies, with each company hiring at least one person -/
def assignGraduates (n k : ℕ) : ℕ :=
  sorry

theorem five_graduates_three_companies : 
  assignGraduates 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_graduates_three_companies_l4074_407455


namespace NUMINAMATH_CALUDE_grouping_ways_correct_l4074_407408

/-- The number of ways to place 4 men and 5 women into three groups. -/
def groupingWays : ℕ := 360

/-- The total number of men. -/
def numMen : ℕ := 4

/-- The total number of women. -/
def numWomen : ℕ := 5

/-- The number of groups. -/
def numGroups : ℕ := 3

/-- The size of each group. -/
def groupSize : ℕ := 3

/-- Predicate to check if a group composition is valid. -/
def validGroup (men women : ℕ) : Prop :=
  men > 0 ∧ women > 0 ∧ men + women = groupSize

/-- The theorem stating the number of ways to group people. -/
theorem grouping_ways_correct :
  ∃ (g1_men g1_women g2_men g2_women g3_men g3_women : ℕ),
    validGroup g1_men g1_women ∧
    validGroup g2_men g2_women ∧
    validGroup g3_men g3_women ∧
    g1_men + g2_men + g3_men = numMen ∧
    g1_women + g2_women + g3_women = numWomen ∧
    groupingWays = 360 :=
  sorry

end NUMINAMATH_CALUDE_grouping_ways_correct_l4074_407408


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l4074_407490

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- For the function h(x) = 5x - 7, h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l4074_407490


namespace NUMINAMATH_CALUDE_radius_ef_is_sqrt_136_l4074_407467

/-- Triangle DEF with semicircles on its sides -/
structure TriangleWithSemicircles where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DEF is a right triangle -/
  right_angle : de^2 + df^2 = ef^2
  /-- Area of semicircle on DE -/
  area_de : (1/2) * Real.pi * (de/2)^2 = 18 * Real.pi
  /-- Arc length of semicircle on DF -/
  arc_df : Real.pi * (df/2) = 10 * Real.pi

/-- The radius of the semicircle on EF is √136 -/
theorem radius_ef_is_sqrt_136 (t : TriangleWithSemicircles) : ef/2 = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_radius_ef_is_sqrt_136_l4074_407467


namespace NUMINAMATH_CALUDE_adjacent_rectangle_area_l4074_407466

-- Define the structure of our rectangle
structure DividedRectangle where
  total_length : ℝ
  total_width : ℝ
  length_split : ℝ -- Point where length is split
  width_split : ℝ -- Point where width is split
  inner_split : ℝ -- Point where the largest rectangle is split

-- Define our specific rectangle
def our_rectangle : DividedRectangle where
  total_length := 5
  total_width := 13
  length_split := 3
  width_split := 9
  inner_split := 4

-- Define areas of known rectangles
def area1 : ℝ := 12
def area2 : ℝ := 15
def area3 : ℝ := 20
def area4 : ℝ := 18
def inner_area : ℝ := 8

-- Theorem to prove
theorem adjacent_rectangle_area (r : DividedRectangle) :
  r.length_split * r.inner_split = area3 - inner_area ∧
  (r.total_length - r.length_split) * r.inner_split = inner_area ∧
  (r.total_length - r.length_split) * (r.total_width - r.width_split) = area4 →
  area4 = 18 := by sorry

end NUMINAMATH_CALUDE_adjacent_rectangle_area_l4074_407466


namespace NUMINAMATH_CALUDE_exists_special_number_l4074_407432

def is_ten_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 →
    (n / 10^d₁ % 10 = n / 10^d₂ % 10) → d₁ = d₂

theorem exists_special_number :
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    is_ten_digit ((10000 * a + 1111 * b)^2 - 1) ∧
    all_digits_distinct ((10000 * a + 1111 * b)^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_special_number_l4074_407432


namespace NUMINAMATH_CALUDE_zekes_estimate_l4074_407401

theorem zekes_estimate (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) (h : x > 2*y) :
  (x + k) - 2*(y + k) < x - 2*y := by
sorry

end NUMINAMATH_CALUDE_zekes_estimate_l4074_407401


namespace NUMINAMATH_CALUDE_power_mod_twenty_l4074_407436

theorem power_mod_twenty : 17^2037 % 20 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_twenty_l4074_407436


namespace NUMINAMATH_CALUDE_total_carrots_grown_l4074_407407

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l4074_407407


namespace NUMINAMATH_CALUDE_hidden_number_l4074_407443

theorem hidden_number (x : ℝ) (hidden : ℝ) : 
  x = -1 → (2 + hidden * x) / 3 = -1 → hidden = 5 := by
  sorry

end NUMINAMATH_CALUDE_hidden_number_l4074_407443


namespace NUMINAMATH_CALUDE_count_ways_2016_l4074_407404

/-- The number of ways to write 2016 as the sum of twos and threes, ignoring order -/
def ways_to_write_2016 : ℕ :=
  (Finset.range 337).card

/-- The theorem stating that there are 337 ways to write 2016 as the sum of twos and threes -/
theorem count_ways_2016 : ways_to_write_2016 = 337 := by
  sorry

end NUMINAMATH_CALUDE_count_ways_2016_l4074_407404


namespace NUMINAMATH_CALUDE_product_101_squared_l4074_407434

theorem product_101_squared : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_product_101_squared_l4074_407434


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_six_l4074_407465

theorem sum_of_roots_equals_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_six_l4074_407465


namespace NUMINAMATH_CALUDE_games_calculation_l4074_407433

def football_games : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games : List Nat := [17, 21, 14, 32, 22, 27]

def total_games : Nat := football_games.sum + baseball_games.sum + basketball_games.sum

def average_games : Nat := total_games / 6

theorem games_calculation :
  total_games = 486 ∧ average_games = 81 := by
  sorry

end NUMINAMATH_CALUDE_games_calculation_l4074_407433


namespace NUMINAMATH_CALUDE_pet_store_dogs_l4074_407422

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:5 and there are 18 cats, prove there are 30 dogs -/
theorem pet_store_dogs :
  let cat_ratio : ℕ := 3
  let dog_ratio : ℕ := 5
  let num_cats : ℕ := 18
  calculate_dogs cat_ratio dog_ratio num_cats = 30 := by
  sorry

#eval calculate_dogs 3 5 18

end NUMINAMATH_CALUDE_pet_store_dogs_l4074_407422


namespace NUMINAMATH_CALUDE_evening_ticket_price_l4074_407468

/-- The price of a matinee ticket in dollars -/
def matinee_price : ℚ := 5

/-- The price of a 3D ticket in dollars -/
def three_d_price : ℚ := 20

/-- The number of matinee tickets sold -/
def matinee_count : ℕ := 200

/-- The number of evening tickets sold -/
def evening_count : ℕ := 300

/-- The number of 3D tickets sold -/
def three_d_count : ℕ := 100

/-- The total revenue in dollars -/
def total_revenue : ℚ := 6600

/-- The price of an evening ticket in dollars -/
def evening_price : ℚ := 12

theorem evening_ticket_price :
  matinee_price * matinee_count + evening_price * evening_count + three_d_price * three_d_count = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_l4074_407468


namespace NUMINAMATH_CALUDE_sector_central_angle_l4074_407439

theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  let θ := (p - 2 * r) / r
  θ = 2.5 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4074_407439


namespace NUMINAMATH_CALUDE_probability_both_divisible_by_4_l4074_407492

/-- A fair 8-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The probability of an event occurring when tossing a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / EightSidedDie.card

/-- The set of outcomes divisible by 4 on an 8-sided die -/
def divisibleBy4 : Finset ℕ := Finset.filter (·.mod 4 = 0) EightSidedDie

/-- The probability of getting a number divisible by 4 on one 8-sided die -/
def probDivisibleBy4 : ℚ := prob divisibleBy4

theorem probability_both_divisible_by_4 :
  probDivisibleBy4 * probDivisibleBy4 = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_both_divisible_by_4_l4074_407492


namespace NUMINAMATH_CALUDE_fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l4074_407449

-- Define a folding operation
def fold (m : ℝ) (x : ℝ) : ℝ := 2 * m - x

-- Theorem 1
theorem fold_three_to_negative_three :
  fold 0 3 = -3 :=
sorry

-- Theorem 2
theorem fold_seven_to_negative_five :
  fold 1 7 = -5 :=
sorry

-- Theorem 3
theorem fold_points_with_distance (m : ℝ) (h : m > 0) :
  ∃ (a b : ℝ), a < b ∧ b - a = m ∧ fold ((a + b) / 2) a = b ∧ a = -(1/2) * m + 1 ∧ b = (1/2) * m + 1 :=
sorry

end NUMINAMATH_CALUDE_fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l4074_407449


namespace NUMINAMATH_CALUDE_f_upper_bound_f_negative_l4074_407444

/-- The function f(x) = ax^2 - (a+1)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- Theorem stating the range of a for which f(x) ≤ 2 for all x in ℝ -/
theorem f_upper_bound (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ -3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2 :=
sorry

/-- Theorem describing the solution set of f(x) < 0 for different ranges of a -/
theorem f_negative (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1) ∨
     (a < 0 ∧ ((x < 1/a) ∨ (x > 1))))) :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_f_negative_l4074_407444


namespace NUMINAMATH_CALUDE_product_equals_533_l4074_407462

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [2, 1, 1, 1]

theorem product_equals_533 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 533 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_533_l4074_407462


namespace NUMINAMATH_CALUDE_gray_opposite_black_l4074_407440

-- Define the colors
inductive Color
| A -- Aqua
| B -- Black
| C -- Crimson
| D -- Dark Blue
| E -- Emerald
| F -- Fuchsia
| G -- Gray
| H -- Hazel

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  adjacent : Color → Color → Prop
  opposite : Color → Color → Prop

-- Define the problem conditions
axiom cube_has_eight_faces : ∀ (c : Cube), c.faces.length = 8

axiom aqua_adjacent_to_dark_blue_and_emerald : 
  ∀ (c : Cube), c.adjacent Color.A Color.D ∧ c.adjacent Color.A Color.E

-- The theorem to prove
theorem gray_opposite_black (c : Cube) : c.opposite Color.G Color.B := by
  sorry


end NUMINAMATH_CALUDE_gray_opposite_black_l4074_407440


namespace NUMINAMATH_CALUDE_largest_divisible_by_digits_correct_l4074_407474

/-- A function that returns true if n is divisible by all of its distinct, non-zero digits -/
def divisible_by_digits (n : ℕ) : Bool :=
  let digits := n.digits 10
  digits.all (λ d => d ≠ 0 ∧ n % d = 0)

/-- The largest three-digit number divisible by all its distinct, non-zero digits -/
def largest_divisible_by_digits : ℕ := 936

theorem largest_divisible_by_digits_correct :
  (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ divisible_by_digits n → n ≤ largest_divisible_by_digits) ∧
  divisible_by_digits largest_divisible_by_digits :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_digits_correct_l4074_407474


namespace NUMINAMATH_CALUDE_equation_solutions_l4074_407405

theorem equation_solutions (x y z v : ℤ) : 
  (x^2 + y^2 + z^2 = 2*x*y*z ↔ x = 0 ∧ y = 0 ∧ z = 0) ∧
  (x^2 + y^2 + z^2 + v^2 = 2*x*y*z*v ↔ x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4074_407405


namespace NUMINAMATH_CALUDE_triangle_classification_l4074_407402

theorem triangle_classification (a b c : ℝ) (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) :
  a = b ∨ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_classification_l4074_407402


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l4074_407477

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l4074_407477


namespace NUMINAMATH_CALUDE_problem_solution_l4074_407489

theorem problem_solution (c d : ℝ) 
  (eq1 : 5 + c = 3 - d)
  (eq2 : 3 + d = 8 + c)
  (eq3 : c - d = 2) : 
  5 - c = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4074_407489


namespace NUMINAMATH_CALUDE_total_money_is_102_l4074_407478

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def anna_money (jack : ℕ) : ℕ := jack * 2

def total_money (eric ben jack anna : ℕ) : ℕ := eric + ben + jack + anna

theorem total_money_is_102 :
  total_money (eric_money (ben_money jack_money)) (ben_money jack_money) jack_money (anna_money jack_money) = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_102_l4074_407478


namespace NUMINAMATH_CALUDE_largest_k_for_real_root_l4074_407446

/-- The quadratic function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + (k-1)^2

/-- The discriminant of f(x) as a function of k -/
def discriminant (k : ℝ) : ℝ := (-k)^2 - 4*(k-1)^2

/-- Theorem: The largest possible real value of k such that f has at least one real root is 2 -/
theorem largest_k_for_real_root :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) → k ≤ 2 ∧ 
  ∃ x : ℝ, f 2 x = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_real_root_l4074_407446


namespace NUMINAMATH_CALUDE_line_perp_para_implies_plane_perp_l4074_407486

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (planePara : Plane → Plane → Prop)
variable (planePerpDir : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_para_implies_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp m α)
  (h2 : para m β) :
  planePerpDir α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_para_implies_plane_perp_l4074_407486


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_2018_l4074_407498

theorem sum_of_x_and_y_is_2018 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x^4 - 2018*x^3 - 2018*y^2*x = y^4 - 2018*y^3 - 2018*y*x^2) : 
  x + y = 2018 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_2018_l4074_407498


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l4074_407441

theorem algebraic_expression_simplification (a : ℝ) :
  a = 2 * Real.sin (60 * π / 180) + 3 →
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6*a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l4074_407441


namespace NUMINAMATH_CALUDE_total_trees_is_fifteen_l4074_407409

/-- The number of apple trees Ava planted -/
def ava_trees : ℕ := 9

/-- The difference between Ava's and Lily's trees -/
def difference : ℕ := 3

/-- The number of apple trees Lily planted -/
def lily_trees : ℕ := ava_trees - difference

/-- The total number of apple trees planted by Ava and Lily -/
def total_trees : ℕ := ava_trees + lily_trees

theorem total_trees_is_fifteen : total_trees = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_is_fifteen_l4074_407409


namespace NUMINAMATH_CALUDE_cube_iff_greater_l4074_407417

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_iff_greater_l4074_407417


namespace NUMINAMATH_CALUDE_rectangular_field_area_l4074_407484

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 105 rupees at 25 paise per meter has an area of 10800 square meters -/
theorem rectangular_field_area (length width : ℝ) (fencing_cost : ℝ) : 
  length / width = 4 / 3 →
  fencing_cost = 105 →
  (2 * (length + width)) * 0.25 = fencing_cost * 100 →
  length * width = 10800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l4074_407484


namespace NUMINAMATH_CALUDE_yellow_candy_bounds_l4074_407463

/-- Represents the state of the candy game -/
structure CandyGame where
  total : ℕ
  yellow : ℕ
  colors : ℕ
  yi_turn : Bool

/-- Defines the rules of the candy game -/
def valid_game (game : CandyGame) : Prop :=
  game.total = 22 ∧
  game.colors = 4 ∧
  game.yellow ≤ game.total ∧
  ∀ other_color, other_color ≠ game.yellow → other_color < game.yellow

/-- Defines a valid move in the game -/
def valid_move (before after : CandyGame) : Prop :=
  (before.yi_turn ∧ 
    ((before.total ≥ 2 ∧ after.total = before.total - 2) ∨ 
     (before.total = 1 ∧ after.total = 0))) ∨
  (¬before.yi_turn ∧ 
    (after.total = before.total - before.colors + 1 ∨ after.total = 0))

/-- Defines the end state of the game -/
def game_end (game : CandyGame) : Prop :=
  game.total = 0

/-- Theorem stating the bounds on the number of yellow candies -/
theorem yellow_candy_bounds (initial : CandyGame) :
  valid_game initial →
  (∃ final : CandyGame, 
    game_end final ∧ 
    (∀ intermediate : CandyGame, valid_move initial intermediate → valid_move intermediate final)) →
  8 ≤ initial.yellow ∧ initial.yellow ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_yellow_candy_bounds_l4074_407463


namespace NUMINAMATH_CALUDE_line_direction_vector_b_l4074_407437

def point_1 : ℝ × ℝ := (-3, 1)
def point_2 : ℝ × ℝ := (1, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (3, b)

theorem line_direction_vector_b (b : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ direction_vector b = k • (point_2 - point_1)) → b = 3 :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_b_l4074_407437


namespace NUMINAMATH_CALUDE_berts_profit_l4074_407491

/-- Calculates the profit for a single item --/
def itemProfit (salesPrice : ℚ) (taxRate : ℚ) : ℚ :=
  salesPrice - (salesPrice * taxRate) - (salesPrice - 10)

/-- Calculates the total profit from the sale --/
def totalProfit (barrelPrice : ℚ) (toolsPrice : ℚ) (fertilizerPrice : ℚ) 
  (barrelTaxRate : ℚ) (toolsTaxRate : ℚ) (fertilizerTaxRate : ℚ) : ℚ :=
  itemProfit barrelPrice barrelTaxRate + 
  itemProfit toolsPrice toolsTaxRate + 
  itemProfit fertilizerPrice fertilizerTaxRate

/-- Theorem stating that Bert's total profit is $14.90 --/
theorem berts_profit : 
  totalProfit 90 50 30 (10/100) (5/100) (12/100) = 149/10 :=
by sorry

end NUMINAMATH_CALUDE_berts_profit_l4074_407491


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4074_407438

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 * i / (1 - 2 * i)
  z = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4074_407438


namespace NUMINAMATH_CALUDE_club_group_size_l4074_407476

theorem club_group_size (N : ℕ) (x : ℕ) 
  (h1 : 10 < N ∧ N < 40)
  (h2 : (N - 3) % 5 = 0 ∧ (N - 3) % 6 = 0)
  (h3 : N % x = 5)
  : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_club_group_size_l4074_407476


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4074_407419

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : IsGeometricSequence a) (h2 : a 4 = 2) :
  a 2 * a 6 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l4074_407419


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l4074_407445

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) : 
  (22 * cost) / (88 * cost) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l4074_407445


namespace NUMINAMATH_CALUDE_multiples_of_five_up_to_hundred_l4074_407415

theorem multiples_of_five_up_to_hundred :
  ∃ n : ℕ, n = 100 ∧ (∃! k : ℕ, k = 20 ∧ (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → (m % 5 = 0 ↔ m ∈ Finset.range k))) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_five_up_to_hundred_l4074_407415


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l4074_407425

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the side length of the small cube removed from each corner -/
def smallCubeSide : ℝ := 4

/-- Calculates the volume of the small cube -/
def smallCubeVolume : ℝ :=
  smallCubeSide ^ 3

/-- The number of corners in a rectangular prism -/
def numCorners : ℕ := 8

/-- Theorem: The percentage of volume removed from the rectangular prism -/
theorem volume_removed_percentage
  (d : PrismDimensions)
  (h1 : d.length = 20)
  (h2 : d.width = 14)
  (h3 : d.height = 12) :
  (numCorners * smallCubeVolume) / (prismVolume d) * 100 = 512 / 3360 * 100 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l4074_407425


namespace NUMINAMATH_CALUDE_sequence_remainder_l4074_407451

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + aₙ) / 2

theorem sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 →
  aₙ = 315 →
  d = 8 →
  aₙ = a₁ + (n - 1) * d →
  (arithmetic_sequence_sum a₁ aₙ n) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_remainder_l4074_407451


namespace NUMINAMATH_CALUDE_max_y_value_l4074_407412

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = (4*x - 5*y)*y) : 
  y ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l4074_407412


namespace NUMINAMATH_CALUDE_h_domain_l4074_407448

def f_domain : Set ℝ := Set.Icc (-3) 6

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3 * x)

theorem h_domain (f : ℝ → ℝ) : 
  {x : ℝ | ∃ y ∈ f_domain, y = -3 * x} = Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_h_domain_l4074_407448


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l4074_407427

theorem greatest_integer_inequality : ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l4074_407427


namespace NUMINAMATH_CALUDE_expression_simplification_l4074_407499

theorem expression_simplification :
  80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4074_407499


namespace NUMINAMATH_CALUDE_smallest_N_with_g_geq_10_N_mod_1000_l4074_407497

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-7 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 7

theorem smallest_N_with_g_geq_10 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N = 610 := by sorry

theorem N_mod_1000 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N ≡ 610 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_smallest_N_with_g_geq_10_N_mod_1000_l4074_407497


namespace NUMINAMATH_CALUDE_first_quartile_of_list_l4074_407411

def list : List ℝ := [42, 24, 30, 28, 26, 19, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (· < m))

theorem first_quartile_of_list : first_quartile list = 25 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_list_l4074_407411


namespace NUMINAMATH_CALUDE_cupcake_price_l4074_407487

/-- Proves that the price of each cupcake is $2 given the problem conditions --/
theorem cupcake_price (cookies_sold : ℕ) (cookie_price : ℚ) (cupcakes_sold : ℕ) 
  (spoons_bought : ℕ) (spoon_price : ℚ) (money_left : ℚ)
  (h1 : cookies_sold = 40)
  (h2 : cookie_price = 4/5)
  (h3 : cupcakes_sold = 30)
  (h4 : spoons_bought = 2)
  (h5 : spoon_price = 13/2)
  (h6 : money_left = 79) :
  let total_earned := cookies_sold * cookie_price + cupcakes_sold * (2 : ℚ)
  let total_spent := spoons_bought * spoon_price + money_left
  total_earned = total_spent := by sorry

end NUMINAMATH_CALUDE_cupcake_price_l4074_407487


namespace NUMINAMATH_CALUDE_tank_capacity_l4074_407479

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  
/-- The tank is 24% full when it contains 72 liters -/
def condition1 (tank : WaterTank) : Prop :=
  0.24 * tank.capacity = 72

/-- The tank is 60% full when it contains 180 liters -/
def condition2 (tank : WaterTank) : Prop :=
  0.60 * tank.capacity = 180

/-- The theorem stating the total capacity of the tank -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : condition1 tank) (h2 : condition2 tank) : 
  tank.capacity = 300 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l4074_407479


namespace NUMINAMATH_CALUDE_unique_solution_system_l4074_407424

theorem unique_solution_system (w x y z : ℝ) :
  w > 0 → x > 0 → y > 0 → z > 0 →
  w + x + y + z = 12 →
  w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27 →
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l4074_407424


namespace NUMINAMATH_CALUDE_boat_license_count_l4074_407488

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or B
  let digit_options := 10  -- 0 to 9
  let digit_positions := 5
  letter_options * digit_options ^ digit_positions

theorem boat_license_count : boat_license_options = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l4074_407488


namespace NUMINAMATH_CALUDE_cyclist_travel_time_l4074_407429

/-- Proves that a cyclist who travels 2.5 miles in 10 minutes will take 20 minutes
    to travel 4 miles when their speed is reduced by 20% due to a headwind -/
theorem cyclist_travel_time (initial_distance : ℝ) (initial_time : ℝ) 
  (new_distance : ℝ) (speed_reduction : ℝ) :
  initial_distance = 2.5 →
  initial_time = 10 →
  new_distance = 4 →
  speed_reduction = 0.2 →
  (new_distance / ((initial_distance / initial_time) * (1 - speed_reduction))) = 20 := by
  sorry

#check cyclist_travel_time

end NUMINAMATH_CALUDE_cyclist_travel_time_l4074_407429


namespace NUMINAMATH_CALUDE_power_division_subtraction_addition_l4074_407430

theorem power_division_subtraction_addition : (-6)^4 / 6^2 - 2^5 + 4^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_division_subtraction_addition_l4074_407430


namespace NUMINAMATH_CALUDE_rat_value_formula_l4074_407400

/-- The number value of a letter in a shifted alphabet with offset N -/
def letterValue (position : ℕ) (N : ℕ) : ℕ := position + N

/-- The sum of letter values for the word "rat" in a shifted alphabet with offset N -/
def ratSum (N : ℕ) : ℕ := letterValue 18 N + letterValue 1 N + letterValue 20 N

/-- The length of the word "rat" -/
def ratLength : ℕ := 3

/-- The number value of the word "rat" in a shifted alphabet with offset N -/
def ratValue (N : ℕ) : ℕ := ratSum N * ratLength

theorem rat_value_formula (N : ℕ) : ratValue N = 117 + 9 * N := by
  sorry

end NUMINAMATH_CALUDE_rat_value_formula_l4074_407400


namespace NUMINAMATH_CALUDE_parabola_translation_up_2_l4074_407494

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (k : ℝ) : Parabola where
  f := λ x => p.f x + k

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola where
  f := λ x => x^2

theorem parabola_translation_up_2 :
  (translate_vertical standard_parabola 2).f = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_up_2_l4074_407494


namespace NUMINAMATH_CALUDE_cos_105_degrees_l4074_407472

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l4074_407472


namespace NUMINAMATH_CALUDE_percentage_difference_l4074_407456

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.25)) :
  y = x * (1 + 0.25) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4074_407456


namespace NUMINAMATH_CALUDE_integral_of_f_l4074_407442

theorem integral_of_f (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) : 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_l4074_407442


namespace NUMINAMATH_CALUDE_min_value_expression_l4074_407459

theorem min_value_expression (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 2) 
  (heq : 2 * a + b - 6 = 0) : 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 1 → y > 2 → 2 * x + y - 6 = 0 → 
    1 / (x - 1) + 2 / (y - 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4074_407459


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l4074_407428

/-- The range of m for which the point P(1-1/3m, m-5) is in the third quadrant --/
theorem point_in_third_quadrant (m : ℝ) : 
  (1 - 1/3*m < 0 ∧ m - 5 < 0) ↔ (3 < m ∧ m < 5) := by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l4074_407428


namespace NUMINAMATH_CALUDE_total_girls_count_l4074_407475

theorem total_girls_count (van1_students van2_students van3_students van4_students van5_students : Nat)
                          (van1_boys van2_boys van3_boys van4_boys van5_boys : Nat)
                          (h1 : van1_students = 24) (h2 : van2_students = 30) (h3 : van3_students = 20)
                          (h4 : van4_students = 36) (h5 : van5_students = 29)
                          (h6 : van1_boys = 12) (h7 : van2_boys = 16) (h8 : van3_boys = 10)
                          (h9 : van4_boys = 18) (h10 : van5_boys = 8) :
  (van1_students - van1_boys) + (van2_students - van2_boys) + (van3_students - van3_boys) +
  (van4_students - van4_boys) + (van5_students - van5_boys) = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_girls_count_l4074_407475


namespace NUMINAMATH_CALUDE_calculation_proof_l4074_407496

theorem calculation_proof :
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4074_407496


namespace NUMINAMATH_CALUDE_smallest_difference_l4074_407418

def Digits : Finset Nat := {1, 3, 4, 6, 7, 8}

def is_valid_subtraction (a b : Nat) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Digits.card = 6) ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 6) ∧
  (∀ d ∈ Digits, (d ∈ a.digits 10 ∨ d ∈ b.digits 10)) ∧
  (∀ d ∈ (a.digits 10 ∪ b.digits 10), d ∈ Digits)

theorem smallest_difference : 
  ∀ a b : Nat, is_valid_subtraction a b → a - b ≥ 473 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l4074_407418


namespace NUMINAMATH_CALUDE_basketball_handshakes_l4074_407482

theorem basketball_handshakes :
  let team_size : ℕ := 5
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (team_size * num_teams) * num_referees
  inter_team_handshakes + player_referee_handshakes = 55 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l4074_407482


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4074_407470

/-- Given a parabola and a hyperbola with an intersection point on the hyperbola's asymptote,
    prove that the hyperbola's eccentricity is √5 under specific conditions. -/
theorem hyperbola_eccentricity (p a b : ℝ) (h1 : p > 0) (h2 : a > 0) (h3 : b > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b/a)*x}
  ∃ A : ℝ × ℝ, A ∈ C₁ ∧ A ∈ C₂ ∧ A ∈ asymptote ∧ 
    (let (x, y) := A
     x - p/2 = p) →
  (Real.sqrt ((a^2 + b^2) / a^2) : ℝ) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4074_407470


namespace NUMINAMATH_CALUDE_circles_intersection_properties_l4074_407453

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y = 0

-- Define the perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Define a point P on circle O1
def P : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the distance from a point to a line
def distance_to_line (p : ℝ × ℝ) (l : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem circles_intersection_properties :
  (∀ x y, line_AB x y ↔ x = y) ∧
  (∀ x y, perp_bisector_AB x y ↔ x + y = 1) ∧
  (∃ P, circle_O1 P.1 P.2 ∧ 
    distance_to_line P line_AB = Real.sqrt 2 / 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_properties_l4074_407453


namespace NUMINAMATH_CALUDE_chicken_egg_production_l4074_407483

/-- Given that 6 chickens lay 30 eggs in 5 days, prove that 10 chickens will lay 80 eggs in 8 days. -/
theorem chicken_egg_production 
  (initial_chickens : ℕ) 
  (initial_eggs : ℕ) 
  (initial_days : ℕ)
  (new_chickens : ℕ) 
  (new_days : ℕ)
  (h1 : initial_chickens = 6)
  (h2 : initial_eggs = 30)
  (h3 : initial_days = 5)
  (h4 : new_chickens = 10)
  (h5 : new_days = 8) :
  (new_chickens * new_days * initial_eggs) / (initial_chickens * initial_days) = 80 :=
by sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l4074_407483


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4074_407414

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x * x = 5 ∧ y * y = 7 ∧
    x + 1/x + y + 1/y = (a * x + b * y) / c ∧
    ∀ (a' b' c' : ℕ+), 
      (∃ (x' y' : ℝ), x' * x' = 5 ∧ y' * y' = 7 ∧
        x' + 1/x' + y' + 1/y' = (a' * x' + b' * y') / c') →
      c ≤ c') →
  a + b + c = 117 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4074_407414


namespace NUMINAMATH_CALUDE_intersection_points_determine_a_l4074_407480

def curve_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = a^2 ∧ 0 ≤ x ∧ x ≤ a

def curve_C₂ (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

theorem intersection_points_determine_a :
  ∀ a : ℝ, a > 0 →
  ∃ A B : ℝ × ℝ,
    curve_C₁ a A.1 A.2 ∧
    curve_C₁ a B.1 B.2 ∧
    curve_C₂ A.1 A.2 ∧
    curve_C₂ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * Real.sqrt 2 / 3)^2 →
    a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_determine_a_l4074_407480


namespace NUMINAMATH_CALUDE_tea_mixture_price_l4074_407426

theorem tea_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  mixture_price = 153 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  ∃ price3 : ℚ,
    price3 = 175.5 ∧
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l4074_407426


namespace NUMINAMATH_CALUDE_pie_eating_contest_l4074_407471

theorem pie_eating_contest (erik_pie frank_pie : ℚ) : 
  erik_pie = 0.6666666666666666 →
  erik_pie = frank_pie + 0.3333333333333333 →
  frank_pie = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l4074_407471


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l4074_407460

/-- Given a principal amount, final amount, and time period, 
    calculate the simple interest rate as a percentage. -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 7.6% -/
theorem simple_interest_rate_example : 
  simple_interest_rate 25000 34500 5 = 76/10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l4074_407460


namespace NUMINAMATH_CALUDE_hundred_from_twos_l4074_407454

theorem hundred_from_twos : (222 / 2) - (22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_from_twos_l4074_407454


namespace NUMINAMATH_CALUDE_milk_cartons_calculation_correct_l4074_407403

/-- Calculates the number of 1L milk cartons needed for lasagna -/
def milk_cartons_needed (servings_per_person : ℕ) : ℕ :=
  let ml_per_cup : ℕ := 250
  let cups_per_serving : ℚ := 1/2
  let people : ℕ := 8
  let total_ml : ℕ := (servings_per_person * people * ml_per_cup * cups_per_serving).ceil.toNat
  (total_ml + 999) / 1000

/-- Theorem: The number of 1L milk cartons needed is correct -/
theorem milk_cartons_calculation_correct (servings_per_person : ℕ) :
  milk_cartons_needed servings_per_person =
  ((8 * servings_per_person : ℚ) * (1/2) * 250 / 1000).ceil.toNat :=
by sorry

end NUMINAMATH_CALUDE_milk_cartons_calculation_correct_l4074_407403


namespace NUMINAMATH_CALUDE_cherries_refund_l4074_407416

def grapes_cost : ℚ := 12.08
def total_spent : ℚ := 2.23

theorem cherries_refund :
  grapes_cost - total_spent = 9.85 := by sorry

end NUMINAMATH_CALUDE_cherries_refund_l4074_407416


namespace NUMINAMATH_CALUDE_sequence_general_term_l4074_407431

/-- Given a sequence {a_n} where S_n is the sum of its first n terms and S_n = 1 - (2/3)a_n,
    prove that the general term a_n is equal to (3/5) * (2/5)^(n-1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 1 - (2/3) * a n) :
    ∀ n, a n = (3/5) * (2/5)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l4074_407431


namespace NUMINAMATH_CALUDE_tan_240_plus_sin_neg_420_l4074_407406

theorem tan_240_plus_sin_neg_420 :
  Real.tan (240 * π / 180) + Real.sin ((-420) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_plus_sin_neg_420_l4074_407406


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4074_407481

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / 2 : ℚ) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4074_407481


namespace NUMINAMATH_CALUDE_power_mod_prime_l4074_407464

theorem power_mod_prime (p : Nat) (h : p.Prime) :
  (3 : ZMod p)^2020 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_power_mod_prime_l4074_407464


namespace NUMINAMATH_CALUDE_no_valid_schedule_for_100_l4074_407495

/-- Represents a duty schedule for militia members -/
structure DutySchedule (n : ℕ) where
  nights : Set (Fin n × Fin n × Fin n)
  all_pairs_once : ∀ i j, i < j → ∃! k, (i, j, k) ∈ nights ∨ (i, k, j) ∈ nights ∨ (j, i, k) ∈ nights ∨ (j, k, i) ∈ nights ∨ (k, i, j) ∈ nights ∨ (k, j, i) ∈ nights

/-- Theorem stating the impossibility of creating a valid duty schedule for 100 militia members -/
theorem no_valid_schedule_for_100 : ¬∃ (schedule : DutySchedule 100), True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_schedule_for_100_l4074_407495


namespace NUMINAMATH_CALUDE_dice_probability_l4074_407457

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def p_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def p_odd : ℚ := 1/2

/-- The number of dice required to show even numbers -/
def num_even : ℕ := 4

/-- The number of dice required to show odd numbers -/
def num_odd : ℕ := 4

theorem dice_probability : 
  (Nat.choose num_dice num_even : ℚ) * p_even ^ num_dice = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l4074_407457


namespace NUMINAMATH_CALUDE_shooter_probabilities_l4074_407485

-- Define the probability of hitting the target on a single shot
def p_hit : ℝ := 0.9

-- Define the number of shots
def n_shots : ℕ := 4

-- Statement 1: Probability of hitting the target on the third shot
def statement1 : Prop := p_hit = 0.9

-- Statement 2: Probability of hitting the target exactly three times
def statement2 : Prop := Nat.choose n_shots 3 * p_hit^3 * (1 - p_hit) = p_hit^3 * (1 - p_hit)

-- Statement 3: Probability of hitting the target at least once
def statement3 : Prop := 1 - (1 - p_hit)^n_shots = 1 - (1 - 0.9)^4

theorem shooter_probabilities :
  statement1 ∧ ¬statement2 ∧ statement3 :=
sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l4074_407485


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l4074_407423

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 5*2 - m = 0) → 
  ∃ (t : ℝ), t ≠ 2 ∧ t^2 - 5*t - m = 0 ∧ t = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_other_root_of_quadratic_l4074_407423


namespace NUMINAMATH_CALUDE_stadium_width_l4074_407410

theorem stadium_width (length height diagonal : ℝ) 
  (h_length : length = 24)
  (h_height : height = 16)
  (h_diagonal : diagonal = 34) :
  ∃ width : ℝ, width = 18 ∧ diagonal^2 = length^2 + width^2 + height^2 := by
sorry

end NUMINAMATH_CALUDE_stadium_width_l4074_407410


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4074_407473

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4074_407473


namespace NUMINAMATH_CALUDE_simplify_radical_fraction_l4074_407447

theorem simplify_radical_fraction (x : ℝ) (h1 : x < 0) :
  ((-x^3).sqrt / x) = -(-x).sqrt := by sorry

end NUMINAMATH_CALUDE_simplify_radical_fraction_l4074_407447


namespace NUMINAMATH_CALUDE_speed_in_still_water_l4074_407413

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l4074_407413


namespace NUMINAMATH_CALUDE_janet_friday_gym_hours_l4074_407469

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_friday_equal : Bool

/-- Theorem: Janet spends 1 hour at the gym on Friday -/
theorem janet_friday_gym_hours (schedule : GymSchedule) 
  (h1 : schedule.total_hours = 5)
  (h2 : schedule.monday_hours = 1.5)
  (h3 : schedule.wednesday_hours = 1.5)
  (h4 : schedule.tuesday_friday_equal = true) :
  ∃ friday_hours : ℝ, friday_hours = 1 ∧ 
  schedule.total_hours = schedule.monday_hours + schedule.wednesday_hours + 2 * friday_hours :=
by
  sorry

end NUMINAMATH_CALUDE_janet_friday_gym_hours_l4074_407469


namespace NUMINAMATH_CALUDE_exam_score_calculation_l4074_407461

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : total_score = 130)
  (h4 : correct_answers = 36) :
  (correct_score * correct_answers - total_score) / (total_questions - correct_answers) = 1 := by
sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l4074_407461


namespace NUMINAMATH_CALUDE_sqrt_pattern_l4074_407452

theorem sqrt_pattern (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l4074_407452


namespace NUMINAMATH_CALUDE_valentines_given_to_children_l4074_407493

theorem valentines_given_to_children (initial : ℕ) (remaining : ℕ) :
  initial = 30 → remaining = 22 → initial - remaining = 8 := by
  sorry

end NUMINAMATH_CALUDE_valentines_given_to_children_l4074_407493


namespace NUMINAMATH_CALUDE_total_crayons_calculation_l4074_407458

/-- Given a group of children where each child has a certain number of crayons,
    calculate the total number of crayons. -/
def total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : ℕ :=
  crayons_per_child * num_children

/-- Theorem stating that the total number of crayons is 648 when each child has 18 crayons
    and there are 36 children. -/
theorem total_crayons_calculation :
  total_crayons 18 36 = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_calculation_l4074_407458
