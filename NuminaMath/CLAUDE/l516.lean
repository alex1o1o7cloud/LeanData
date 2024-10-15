import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l516_51675

/-- The value of m for which the ellipse 3x^2 + 9y^2 = 9 and 
    the hyperbola (x-2)^2 - m(y+1)^2 = 1 are tangent -/
theorem ellipse_hyperbola_tangent : 
  ∃! m : ℝ, ∀ x y : ℝ, 
    (3 * x^2 + 9 * y^2 = 9 ∧ (x - 2)^2 - m * (y + 1)^2 = 1) →
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ 
      3 * p.1^2 + 9 * p.2^2 = 9 ∧ 
      (p.1 - 2)^2 - m * (p.2 + 1)^2 = 1) →
    m = 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l516_51675


namespace NUMINAMATH_CALUDE_union_M_N_l516_51611

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1}

theorem union_M_N : M ∪ N = {x | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l516_51611


namespace NUMINAMATH_CALUDE_hundredth_odd_integer_and_divisibility_l516_51683

theorem hundredth_odd_integer_and_divisibility :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧ ¬(199 % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_integer_and_divisibility_l516_51683


namespace NUMINAMATH_CALUDE_candy_distribution_l516_51691

def candies_for_child (n : ℕ) : ℕ := 2^(n - 1)

def total_candies (n : ℕ) : ℕ := 2^n - 1

theorem candy_distribution (total : ℕ) (h : total = 2007) :
  let n := (Nat.log 2 (total + 1)).succ
  (total_candies n - total, n) = (40, 11) := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l516_51691


namespace NUMINAMATH_CALUDE_warehouse_storage_problem_l516_51644

/-- Represents the warehouse storage problem -/
theorem warehouse_storage_problem 
  (second_floor_space : ℝ) 
  (h1 : second_floor_space > 0) 
  (h2 : 3 * second_floor_space - (1/4) * second_floor_space = 55000) : 
  (1/4) * second_floor_space = 5000 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_storage_problem_l516_51644


namespace NUMINAMATH_CALUDE_three_from_eight_l516_51634

theorem three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_from_eight_l516_51634


namespace NUMINAMATH_CALUDE_min_difference_of_roots_l516_51607

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x - t else 2 * (x + 1) - t

theorem min_difference_of_roots (t : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f t x₁ = 0 ∧ f t x₂ = 0 →
  ∃ min_diff : ℝ, (∀ y₁ y₂ : ℝ, y₁ > y₂ → f t y₁ = 0 → f t y₂ = 0 → y₁ - y₂ ≥ min_diff) ∧
               min_diff = 15/16 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_roots_l516_51607


namespace NUMINAMATH_CALUDE_exists_function_sum_one_not_exists_function_diff_one_l516_51626

-- Part a
theorem exists_function_sum_one : 
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

-- Part b
theorem not_exists_function_diff_one : 
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) - f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

end NUMINAMATH_CALUDE_exists_function_sum_one_not_exists_function_diff_one_l516_51626


namespace NUMINAMATH_CALUDE_probability_all_digits_different_l516_51610

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_three_digit_same_digits : ℕ := 9

theorem probability_all_digits_different :
  (count_three_digit_numbers - count_three_digit_same_digits : ℚ) / count_three_digit_numbers = 99/100 :=
sorry

end NUMINAMATH_CALUDE_probability_all_digits_different_l516_51610


namespace NUMINAMATH_CALUDE_inverse_inequality_l516_51640

theorem inverse_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l516_51640


namespace NUMINAMATH_CALUDE_eddie_pies_l516_51656

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The total number of pies they can bake in 7 days -/
def total_pies : ℕ := 119

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- Eddie can bake 3 pies a day -/
theorem eddie_pies : ∃ (eddie_pies : ℕ), 
  eddie_pies = 3 ∧ 
  days * (eddie_pies + sister_pies + mother_pies) = total_pies := by
  sorry

end NUMINAMATH_CALUDE_eddie_pies_l516_51656


namespace NUMINAMATH_CALUDE_solve_for_y_l516_51681

theorem solve_for_y (x y : ℚ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l516_51681


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l516_51609

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chemistry_mathematics (P C M : ℝ)
  (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l516_51609


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l516_51631

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_equivalence 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l516_51631


namespace NUMINAMATH_CALUDE_rubber_duck_race_l516_51698

theorem rubber_duck_race (regular_price : ℚ) (large_price : ℚ) (regular_sold : ℕ) (total_raised : ℚ) :
  regular_price = 3 →
  large_price = 5 →
  regular_sold = 221 →
  total_raised = 1588 →
  ∃ (large_sold : ℕ), large_sold = 185 ∧ 
    regular_price * regular_sold + large_price * large_sold = total_raised :=
by sorry

end NUMINAMATH_CALUDE_rubber_duck_race_l516_51698


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l516_51665

theorem difference_of_squares_factorization (a : ℝ) : a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l516_51665


namespace NUMINAMATH_CALUDE_ways_1800_eq_partitions_300_l516_51668

/-- The number of ways to write a positive integer as a sum of ones, twos, and threes, ignoring order -/
def numWays (n : ℕ) : ℕ := sorry

/-- The number of ways to partition a positive integer into four non-negative integer parts -/
def numPartitions4 (n : ℕ) : ℕ := sorry

/-- Theorem stating the equivalence between the two counting problems for n = 1800 -/
theorem ways_1800_eq_partitions_300 : numWays 1800 = numPartitions4 300 := by sorry

end NUMINAMATH_CALUDE_ways_1800_eq_partitions_300_l516_51668


namespace NUMINAMATH_CALUDE_dalton_watched_nine_movies_l516_51682

/-- The number of movies watched by Dalton in the Superhero Fan Club -/
def dalton_movies : ℕ := sorry

/-- The number of movies watched by Hunter in the Superhero Fan Club -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex in the Superhero Fan Club -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three members -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched by the Superhero Fan Club -/
def total_different_movies : ℕ := 30

theorem dalton_watched_nine_movies :
  dalton_movies + hunter_movies + alex_movies - 3 * movies_watched_together = total_different_movies ∧
  dalton_movies = 9 := by sorry

end NUMINAMATH_CALUDE_dalton_watched_nine_movies_l516_51682


namespace NUMINAMATH_CALUDE_inequality_proof_l516_51659

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l516_51659


namespace NUMINAMATH_CALUDE_permutations_formula_l516_51696

-- Define the number of permutations
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem permutations_formula {n k : ℕ} (h : 1 ≤ k ∧ k ≤ n) :
  permutations n k = (Nat.factorial n) / (Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_permutations_formula_l516_51696


namespace NUMINAMATH_CALUDE_total_values_count_l516_51601

theorem total_values_count (initial_mean correct_mean : ℝ) 
  (incorrect_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 150 →
  correct_mean = 151.25 →
  incorrect_value = 135 →
  correct_value = 160 →
  (n : ℝ) * initial_mean = (n : ℝ) * correct_mean - (correct_value - incorrect_value) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_total_values_count_l516_51601


namespace NUMINAMATH_CALUDE_coexisting_expression_coexisting_negation_l516_51643

/-- Definition of coexisting rational number pairs -/
def is_coexisting (a b : ℚ) : Prop := a * b = a - b - 1

/-- Theorem 1: For coexisting pairs, the given expression equals 1/2 -/
theorem coexisting_expression (a b : ℚ) (h : is_coexisting a b) :
  3 * a * b - a + (1/2) * (a + b - 5 * a * b) + 1 = 1/2 := by sorry

/-- Theorem 2: If (a,b) is coexisting, then (-b,-a) is also coexisting -/
theorem coexisting_negation (a b : ℚ) (h : is_coexisting a b) :
  is_coexisting (-b) (-a) := by sorry

end NUMINAMATH_CALUDE_coexisting_expression_coexisting_negation_l516_51643


namespace NUMINAMATH_CALUDE_rahul_salary_l516_51686

def salary_calculation (salary : ℝ) : ℝ :=
  let after_rent := salary * 0.8
  let after_education := after_rent * 0.9
  let after_clothes := after_education * 0.9
  after_clothes

theorem rahul_salary : ∃ (salary : ℝ), salary_calculation salary = 1377 ∧ salary = 2125 := by
  sorry

end NUMINAMATH_CALUDE_rahul_salary_l516_51686


namespace NUMINAMATH_CALUDE_parabola_directrix_l516_51667

/-- Given a parabola y = -4x^2 + 8x - 1, its directrix is y = 49/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = -4 * x^2 + 8 * x - 1 →
  ∃ (k : ℝ), k = 49/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = -4 * x₀^2 + 8 * x₀ - 1 →
    ∃ (x₁ : ℝ), (x₀ - x₁)^2 + (y₀ - k)^2 = (y₀ - k)^2 / 4) :=
by sorry


end NUMINAMATH_CALUDE_parabola_directrix_l516_51667


namespace NUMINAMATH_CALUDE_two_numbers_difference_l516_51654

theorem two_numbers_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 20000) 
  (h3 : a % 9 = 0 ∨ b % 9 = 0) (h4 : 2 * a + 6 = b) : b - a = 6670 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l516_51654


namespace NUMINAMATH_CALUDE_complete_collection_probability_l516_51622

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := 8

theorem complete_collection_probability :
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_complete_collection_probability_l516_51622


namespace NUMINAMATH_CALUDE_same_heads_probability_l516_51647

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 3

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem same_heads_probability :
  probability = 3 / 32 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_l516_51647


namespace NUMINAMATH_CALUDE_problem_solution_l516_51621

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 24^(2/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l516_51621


namespace NUMINAMATH_CALUDE_solution_characterization_l516_51617

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (0, 0, 1), (2/3, -1/3, 2/3), (0, 1, 0), (1, 0, 0), (-1, 1, 1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 1 ∧
  x^2*y + y^2*z + z^2*x = x*y^2 + y*z^2 + z*x^2 ∧
  x^3 + y^2 + z = y^3 + z^2 + x

theorem solution_characterization :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l516_51617


namespace NUMINAMATH_CALUDE_f_2_equals_100_l516_51676

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_2_equals_100 :
  ∃ y : ℝ, f 5 y = 142 ∧ f 2 y = 100 :=
by sorry

end NUMINAMATH_CALUDE_f_2_equals_100_l516_51676


namespace NUMINAMATH_CALUDE_bottles_per_day_l516_51652

def total_bottles : ℕ := 355
def total_days : ℕ := 71

theorem bottles_per_day : 
  total_bottles / total_days = 5 := by sorry

end NUMINAMATH_CALUDE_bottles_per_day_l516_51652


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_three_eq_sqrt_two_l516_51658

theorem sqrt_six_div_sqrt_three_eq_sqrt_two :
  Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_three_eq_sqrt_two_l516_51658


namespace NUMINAMATH_CALUDE_red_white_red_probability_l516_51635

/-- The probability of drawing a red marble, then a white marble, and finally a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability :
  let total_marbles : ℕ := 10
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_first_red : ℚ := red_marbles / total_marbles
  let prob_second_white : ℚ := white_marbles / (total_marbles - 1)
  let prob_third_red : ℚ := (red_marbles - 1) / (total_marbles - 2)
  prob_first_red * prob_second_white * prob_third_red = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_red_white_red_probability_l516_51635


namespace NUMINAMATH_CALUDE_same_color_socks_probability_l516_51636

def total_red_socks : ℕ := 12
def total_blue_socks : ℕ := 10

theorem same_color_socks_probability :
  let total_socks := total_red_socks + total_blue_socks
  let same_color_combinations := (total_red_socks.choose 2) + (total_blue_socks.choose 2)
  let total_combinations := total_socks.choose 2
  (same_color_combinations : ℚ) / total_combinations = 37 / 77 := by
  sorry

end NUMINAMATH_CALUDE_same_color_socks_probability_l516_51636


namespace NUMINAMATH_CALUDE_rollo_guinea_pig_food_l516_51699

/-- The amount of food needed for Rollo's guinea pigs -/
def guinea_pig_food : ℕ → ℕ
| 1 => 2  -- First guinea pig eats 2 cups
| 2 => 2 * guinea_pig_food 1  -- Second eats twice as much as the first
| 3 => guinea_pig_food 2 + 3  -- Third eats 3 cups more than the second
| _ => 0  -- For completeness, though we only have 3 guinea pigs

/-- The total amount of food needed for all guinea pigs -/
def total_food : ℕ := guinea_pig_food 1 + guinea_pig_food 2 + guinea_pig_food 3

theorem rollo_guinea_pig_food : total_food = 13 := by
  sorry

end NUMINAMATH_CALUDE_rollo_guinea_pig_food_l516_51699


namespace NUMINAMATH_CALUDE_rectangular_room_shorter_side_l516_51614

theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 42)
  (h_area : area = 108)
  (h_rect : ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
            2 * (length + width) = perimeter ∧
            length * width = area) :
  ∃ (shorter_side : ℝ), shorter_side = 9 ∧
    ∃ (longer_side : ℝ), longer_side > shorter_side ∧
      2 * (shorter_side + longer_side) = perimeter ∧
      shorter_side * longer_side = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_room_shorter_side_l516_51614


namespace NUMINAMATH_CALUDE_correspondence_theorem_l516_51697

theorem correspondence_theorem (m n : ℕ) (l : ℕ) 
  (h1 : l ≥ m * (n / 2))
  (h2 : l ≤ n * (m / 2)) :
  l = m * (n / 2) ∧ l = n * (m / 2) :=
sorry

end NUMINAMATH_CALUDE_correspondence_theorem_l516_51697


namespace NUMINAMATH_CALUDE_outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l516_51661

/-- Represents a car driving in a circle -/
structure CircularDrivingCar where
  inner_radius : ℝ
  outer_radius : ℝ
  wheel_distance : ℝ

/-- The properties of the car as described in the problem -/
def problem_car : CircularDrivingCar :=
  { inner_radius := 1.5,  -- This value is derived from the solution, not given directly
    outer_radius := 3,    -- This value is derived from the solution, not given directly
    wheel_distance := 1.5 }

/-- The theorem stating the relationship between the outer and inner wheel speeds -/
theorem outer_wheel_speed_double (car : CircularDrivingCar) :
  car.outer_radius = 2 * car.inner_radius :=
sorry

/-- The theorem stating the relationship between the radii and the wheel distance -/
theorem radii_difference (car : CircularDrivingCar) :
  car.outer_radius - car.inner_radius = car.wheel_distance :=
sorry

/-- The main theorem to prove -/
theorem outer_wheel_circumference (car : CircularDrivingCar) :
  2 * π * car.outer_radius = π * 6 :=
sorry

end NUMINAMATH_CALUDE_outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l516_51661


namespace NUMINAMATH_CALUDE_unique_five_numbers_l516_51620

def triple_sums (a b c d e : ℝ) : List ℝ :=
  [a + b + c, a + b + d, a + b + e, a + c + d, a + c + e, a + d + e,
   b + c + d, b + c + e, b + d + e, c + d + e]

theorem unique_five_numbers :
  ∃! (a b c d e : ℝ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    triple_sums a b c d e = [3, 4, 6, 7, 9, 10, 11, 14, 15, 17] :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_numbers_l516_51620


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l516_51690

-- Define a point on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define a triangle on the grid
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d12 := (t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2
  let d23 := (t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2
  let d31 := (t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

-- Define the five triangles
def triangle1 : Triangle := { v1 := {x := 0, y := 7}, v2 := {x := 2, y := 7}, v3 := {x := 1, y := 5} }
def triangle2 : Triangle := { v1 := {x := 4, y := 3}, v2 := {x := 4, y := 5}, v3 := {x := 6, y := 3} }
def triangle3 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 3, y := 3}, v3 := {x := 6, y := 2} }
def triangle4 : Triangle := { v1 := {x := 1, y := 1}, v2 := {x := 0, y := 3}, v3 := {x := 3, y := 1} }
def triangle5 : Triangle := { v1 := {x := 3, y := 6}, v2 := {x := 4, y := 4}, v3 := {x := 5, y := 7} }

-- Theorem statement
theorem three_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  ¬(isIsosceles triangle4) ∧
  ¬(isIsosceles triangle5) := by
  sorry

end NUMINAMATH_CALUDE_three_isosceles_triangles_l516_51690


namespace NUMINAMATH_CALUDE_apples_given_theorem_l516_51678

/-- The number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

/-- Proof that the number of apples given to Melanie is correct -/
theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h : initial_apples ≥ current_apples) :
  apples_given_to_melanie initial_apples current_apples = initial_apples - current_apples :=
by
  sorry

/-- Verifying the specific case in the problem -/
example : apples_given_to_melanie 43 16 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_given_theorem_l516_51678


namespace NUMINAMATH_CALUDE_jasmine_laps_l516_51651

/-- Calculates the total number of laps swum in a given number of weeks -/
def total_laps (laps_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * num_weeks

/-- Proves that Jasmine swims 300 laps in five weeks -/
theorem jasmine_laps : total_laps 12 5 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_laps_l516_51651


namespace NUMINAMATH_CALUDE_eg_length_l516_51650

/-- A quadrilateral with specific side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The theorem stating the length of EG in the specific quadrilateral -/
theorem eg_length (q : Quadrilateral) 
  (h1 : q.EF = 7)
  (h2 : q.FG = 13)
  (h3 : q.GH = 7)
  (h4 : q.HE = 11) :
  q.EG = 13 := by
  sorry


end NUMINAMATH_CALUDE_eg_length_l516_51650


namespace NUMINAMATH_CALUDE_decrease_interval_of_f_shifted_l516_51630

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the interval of decrease for f(x+1)
def interval_of_decrease : Set ℝ := Set.Ioo 0 2

-- Theorem statement
theorem decrease_interval_of_f_shifted :
  ∀ x ∈ interval_of_decrease, f' (x + 1) < 0 :=
sorry

end NUMINAMATH_CALUDE_decrease_interval_of_f_shifted_l516_51630


namespace NUMINAMATH_CALUDE_max_n_for_T_sum_less_than_2023_l516_51638

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

def c_sequence (n : ℕ) : ℕ := arithmetic_sequence (geometric_sequence n)

def T_sum (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_for_T_sum_less_than_2023 :
  ∀ n : ℕ, T_sum n < 2023 → n ≤ 9 ∧ T_sum 9 < 2023 ∧ T_sum 10 ≥ 2023 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_T_sum_less_than_2023_l516_51638


namespace NUMINAMATH_CALUDE_sams_water_buckets_l516_51632

-- Define the initial amount of water
def initial_water : Real := 1

-- Define the additional amount of water
def additional_water : Real := 8.8

-- Define the total amount of water
def total_water : Real := initial_water + additional_water

-- Theorem statement
theorem sams_water_buckets : total_water = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_sams_water_buckets_l516_51632


namespace NUMINAMATH_CALUDE_max_pieces_theorem_l516_51641

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.eraseDups.length

def max_pieces : ℕ := 7

theorem max_pieces_theorem :
  ∀ n : ℕ, n > max_pieces →
    ¬∃ (A B : ℕ), is_five_digit A ∧ is_five_digit B ∧ has_distinct_digits A ∧ A = B * n :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_theorem_l516_51641


namespace NUMINAMATH_CALUDE_ricks_ironing_rate_l516_51673

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := sorry

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of hours Rick spent ironing dress shirts -/
def shirt_hours : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def pant_hours : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem ricks_ironing_rate :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = total_pieces ∧
  pants_per_hour = 3 :=
sorry

end NUMINAMATH_CALUDE_ricks_ironing_rate_l516_51673


namespace NUMINAMATH_CALUDE_decreasing_interval_minimum_a_l516_51612

noncomputable section

/-- The function f(x) = (2 - a)(x - 1) - 2ln(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

/-- The function g(x) = f(x) + x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

/-- The derivative of g(x) -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 - a - 2 / x

theorem decreasing_interval (a : ℝ) :
  (g' a 1 = -1 ∧ g a 1 = 1) →
  ∀ x, 0 < x → x < 2 → g' a x < 0 :=
sorry

theorem minimum_a :
  (∀ x, 0 < x → x < 1/2 → f a x > 0) →
  a ≥ 2 - 4 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_minimum_a_l516_51612


namespace NUMINAMATH_CALUDE_find_g_of_x_l516_51649

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 - 3 * x + 5 + g x = 3 * x^4 + 7 * x^2 - 2 * x - 4) → 
  (g x = -2 * x^5 + 3 * x^4 - 4 * x^3 + 7 * x^2 - x - 9) := by
sorry

end NUMINAMATH_CALUDE_find_g_of_x_l516_51649


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l516_51688

theorem ellipse_hyperbola_product (a b : ℝ) 
  (h_ellipse : b^2 - a^2 = 25)
  (h_hyperbola : a^2 + b^2 = 64) : 
  |a * b| = Real.sqrt (3461 / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l516_51688


namespace NUMINAMATH_CALUDE_problem_solution_l516_51628

theorem problem_solution (a b : ℝ) (h1 : b - a = -6) (h2 : a * b = 7) :
  a^2 * b - a * b^2 = -42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l516_51628


namespace NUMINAMATH_CALUDE_projection_result_l516_51680

/-- Given two vectors a and b in ℝ², if both are projected onto the same vector v
    resulting in p, then p is equal to (48/53, 168/53). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-2, 4) → 
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ (a - p) • v = 0) → 
  (∃ (k₃ k₄ : ℝ), p = k₃ • v ∧ (b - p) • v = 0) → 
  p = (48/53, 168/53) := by
  sorry

end NUMINAMATH_CALUDE_projection_result_l516_51680


namespace NUMINAMATH_CALUDE_only_extend_line_segment_valid_l516_51663

-- Define the geometric objects
structure StraightLine
structure LineSegment where
  endpoint1 : Point
  endpoint2 : Point
structure Ray where
  endpoint : Point

-- Define the statements
inductive GeometricStatement
  | ExtendStraightLine
  | ExtendLineSegment
  | ExtendRay
  | DrawStraightLineWithLength
  | CutOffSegmentOnRay

-- Define a predicate for valid operations
def is_valid_operation (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.ExtendLineSegment => true
  | _ => false

-- Theorem statement
theorem only_extend_line_segment_valid :
  ∀ s : GeometricStatement, is_valid_operation s ↔ s = GeometricStatement.ExtendLineSegment := by
  sorry

end NUMINAMATH_CALUDE_only_extend_line_segment_valid_l516_51663


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l516_51602

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2 : ℝ) * a * b = 24 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l516_51602


namespace NUMINAMATH_CALUDE_value_of_expression_l516_51627

theorem value_of_expression (x y : ℝ) (h : x - 2*y = 3) : x - 2*y + 4 = 7 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_l516_51627


namespace NUMINAMATH_CALUDE_equation_three_solutions_l516_51606

theorem equation_three_solutions :
  let f : ℝ → ℝ := λ x => (x^2 - 4) * (x^2 - 1) - (x^2 + 3*x + 2) * (x^2 - 8*x + 7)
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l516_51606


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_sequence_satisfies_conditions_l516_51623

/-- The sequence a_n is defined as 10^n + n -/
def a (n : ℕ) : ℕ := 10^n + n

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  a 1 = 11 ∧ a 2 = 102 ∧ a 3 = 1003 ∧ a 4 = 10004 := by
  sorry

/-- The sequence a_n satisfies the given first four terms -/
theorem sequence_satisfies_conditions : ∃ f : ℕ → ℕ, 
  (f 1 = 11 ∧ f 2 = 102 ∧ f 3 = 1003 ∧ f 4 = 10004) ∧
  (∀ n : ℕ, f n = a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_sequence_satisfies_conditions_l516_51623


namespace NUMINAMATH_CALUDE_mod_seven_equality_l516_51655

theorem mod_seven_equality : (47 ^ 2049 - 18 ^ 2049) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_equality_l516_51655


namespace NUMINAMATH_CALUDE_initial_skittles_count_l516_51693

/-- Proves that the initial number of Skittles is equal to the product of the number of friends and the number of Skittles each friend received. -/
theorem initial_skittles_count (num_friends num_skittles_per_friend : ℕ) :
  num_friends * num_skittles_per_friend = num_friends * num_skittles_per_friend :=
by sorry

#check initial_skittles_count 5 8

end NUMINAMATH_CALUDE_initial_skittles_count_l516_51693


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l516_51695

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (↑x : ℚ) * 105 / 100 = 21 ∧ 
  ∀ (n : ℕ+), n < 21 → ¬∃ (y : ℕ+), (↑y : ℚ) * 105 / 100 = ↑n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l516_51695


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l516_51618

theorem no_infinite_sequence_exists : 
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l516_51618


namespace NUMINAMATH_CALUDE_duck_cow_problem_l516_51629

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 40) → C = 20 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l516_51629


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_l516_51664

theorem greatest_five_digit_multiple : ∃ n : ℕ, 
  n ≤ 99999 ∧ 
  n ≥ 10000 ∧
  n % 9 = 0 ∧ 
  n % 6 = 0 ∧ 
  n % 2 = 0 ∧
  ∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 9 = 0 ∧ m % 6 = 0 ∧ m % 2 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_l516_51664


namespace NUMINAMATH_CALUDE_friends_team_assignment_l516_51639

theorem friends_team_assignment :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_assign := num_teams ^ num_friends
  ways_to_assign = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l516_51639


namespace NUMINAMATH_CALUDE_evaluate_F_of_f_l516_51672

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 + 1
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem evaluate_F_of_f : F 4 (f 3) = 140 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_of_f_l516_51672


namespace NUMINAMATH_CALUDE_bucket_fill_time_l516_51605

/-- The time taken to fill a bucket completely, given that two-thirds of it is filled in 90 seconds at a constant rate. -/
theorem bucket_fill_time (fill_rate : ℝ) (h1 : fill_rate > 0) : 
  (2 / 3 : ℝ) / fill_rate = 90 → 1 / fill_rate = 135 := by sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l516_51605


namespace NUMINAMATH_CALUDE_henri_total_miles_l516_51616

-- Define the variables
def gervais_average_miles : ℕ := 315
def gervais_days : ℕ := 3
def additional_miles : ℕ := 305

-- Define the theorem
theorem henri_total_miles :
  let gervais_total := gervais_average_miles * gervais_days
  let henri_total := gervais_total + additional_miles
  henri_total = 1250 := by
  sorry

end NUMINAMATH_CALUDE_henri_total_miles_l516_51616


namespace NUMINAMATH_CALUDE_growth_rate_inequality_l516_51692

theorem growth_rate_inequality (p q x : ℝ) (h : p ≠ q) :
  (1 + x)^2 = (1 + p) * (1 + q) → x < (p + q) / 2 := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_inequality_l516_51692


namespace NUMINAMATH_CALUDE_complement_of_union_l516_51613

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_union :
  (Aᶜ ∩ Bᶜ) ∩ U = {3, 5} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l516_51613


namespace NUMINAMATH_CALUDE_at_least_two_black_balls_count_l516_51604

def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 4
def balls_drawn : ℕ := 4

theorem at_least_two_black_balls_count :
  (Finset.sum (Finset.range 3) (λ i => 
    Nat.choose total_black_balls (i + 2) * Nat.choose total_white_balls (balls_drawn - (i + 2)))) = 115 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_black_balls_count_l516_51604


namespace NUMINAMATH_CALUDE_andrey_gifts_l516_51625

theorem andrey_gifts :
  ∀ (n : ℕ) (a : ℕ),
    n > 2 →
    n * (n - 2) = a * (n - 1) + 16 →
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_andrey_gifts_l516_51625


namespace NUMINAMATH_CALUDE_ghost_paths_count_l516_51660

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways a ghost can enter and exit the mansion -/
def ghost_paths : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that there are exactly 56 ways for a ghost to enter and exit the mansion -/
theorem ghost_paths_count : ghost_paths = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_paths_count_l516_51660


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l516_51619

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = 2x -/
def f : ℝ → ℝ := fun x ↦ 2 * x

/-- Theorem: The function f(x) = 2x is a direct proportion function -/
theorem f_is_direct_proportion : IsDirectProportion f := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l516_51619


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l516_51674

theorem negation_of_quadratic_inequality (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l516_51674


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l516_51603

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l516_51603


namespace NUMINAMATH_CALUDE_triangle_centroid_incenter_relation_l516_51669

open Real

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define functions to calculate centroid and incenter
def centroid (t : Triangle) : ℝ × ℝ := sorry

def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate squared distance between two points
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_centroid_incenter_relation :
  ∃ k : ℝ, ∀ t : Triangle, ∀ P : ℝ × ℝ,
    let G := centroid t
    let I := incenter t
    dist_squared P t.A + dist_squared P t.B + dist_squared P t.C + dist_squared P I =
    k * (dist_squared P G + dist_squared G t.A + dist_squared G t.B + dist_squared G t.C + dist_squared G I) :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_incenter_relation_l516_51669


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l516_51648

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 10 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l516_51648


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_two_l516_51685

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 5/2
  | (n + 2) => 7/2 * G (n + 1) - G n

theorem sum_of_reciprocal_G_powers_of_two : ∑' n, 1 / G (2^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_two_l516_51685


namespace NUMINAMATH_CALUDE_savings_calculation_l516_51646

/-- Calculates a person's savings given their income and the ratio of income to expenditure. -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that for a person with an income of 36000 and an income to expenditure ratio of 9:8, their savings are 4000. -/
theorem savings_calculation :
  calculate_savings 36000 9 8 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l516_51646


namespace NUMINAMATH_CALUDE_simple_random_sampling_problem_l516_51600

/-- Prove that in a simple random sampling where 13 individuals are drawn one by one
    from a group of n individuals (n > 13), if the probability for each of the remaining
    individuals to be drawn on the second draw is 1/3, then n = 37. -/
theorem simple_random_sampling_problem (n : ℕ) (h1 : n > 13) :
  (12 : ℝ) / (n - 1 : ℝ) = (1 : ℝ) / 3 → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_problem_l516_51600


namespace NUMINAMATH_CALUDE_intersection_point_of_parabolas_l516_51633

-- Define the parabolas
def C₁ (x y : ℝ) : Prop :=
  (x - (Real.sqrt 2 - 1))^2 = 2 * (y - 1)^2

def C₂ (a b x y : ℝ) : Prop :=
  x^2 - a*y + x + 2*b = 0

-- Define the perpendicular tangents condition
def perpendicularTangents (a : ℝ) (x y : ℝ) : Prop :=
  (2*y - 2) * (2*y - a) = -1

-- Theorem statement
theorem intersection_point_of_parabolas
  (a b : ℝ) (h : ∃ x y, C₁ x y ∧ C₂ a b x y ∧ perpendicularTangents a x y) :
  ∃ x y, C₁ x y ∧ C₂ a b x y ∧ x = Real.sqrt 2 - 1/2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_parabolas_l516_51633


namespace NUMINAMATH_CALUDE_T_formula_l516_51679

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_T_formula_l516_51679


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l516_51653

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l516_51653


namespace NUMINAMATH_CALUDE_identities_proof_l516_51671

theorem identities_proof (a : ℝ) (n k : ℤ) : 
  ((-a^3 * (-a)^3)^2 + (-a^2 * (-a)^2)^3 = 0) ∧ 
  ((-1:ℝ)^n * a^(n+k) = (-a)^n * a^k) := by
  sorry

end NUMINAMATH_CALUDE_identities_proof_l516_51671


namespace NUMINAMATH_CALUDE_polynomial_characterization_l516_51694

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2 * x * y * z = x + y + z →
    (P x) / (y * z) + (P y) / (z * x) + (P z) / (x * y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_characterization (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (c : ℝ), ∀ (x : ℝ), P x = c * (x^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l516_51694


namespace NUMINAMATH_CALUDE_equation_roots_l516_51608

/-- The equation in question -/
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

/-- The condition for having exactly two distinct complex roots -/
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    equation x₁ k ∧ equation x₂ k ∧
    ∀ x, equation x k → (x = x₁ ∨ x = x₂)

/-- The main theorem -/
theorem equation_roots (k : ℂ) :
  has_two_distinct_roots k ↔ (k = 2*I ∨ k = -2*I) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l516_51608


namespace NUMINAMATH_CALUDE_angle_sum_properties_l516_51662

/-- Given two obtuse angles α and β whose terminal sides intersect the unit circle at points
    with x-coordinates -√2/10 and -2√5/5 respectively, prove that tan(α+β) = -5/3 and α+2β = 9π/4 -/
theorem angle_sum_properties (α β : Real) (hα : α > π/2) (hβ : β > π/2)
  (hA : Real.cos α = -Real.sqrt 2 / 10)
  (hB : Real.cos β = -2 * Real.sqrt 5 / 5) :
  Real.tan (α + β) = -5/3 ∧ α + 2*β = 9*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_properties_l516_51662


namespace NUMINAMATH_CALUDE_difference_of_squares_l516_51624

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (product_eq : x * y = 80) :
  x^2 - y^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l516_51624


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l516_51687

/-- Proves that the interest rate at which a person borrowed money is 4% per annum,
    given the specified conditions. -/
theorem borrowing_interest_rate
  (loan_amount : ℝ)
  (loan_duration : ℕ)
  (lending_rate : ℝ)
  (yearly_gain : ℝ)
  (h1 : loan_amount = 7000)
  (h2 : loan_duration = 2)
  (h3 : lending_rate = 0.06)
  (h4 : yearly_gain = 140)
  : ∃ (borrowing_rate : ℝ), borrowing_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_borrowing_interest_rate_l516_51687


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l516_51666

/-- A triangle with sides a, b, and c is obtuse if and only if 
    the square of the longest side is greater than the sum of 
    squares of the other two sides. -/
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ a^2 + b^2 < c^2) ∨
  (a ≤ c ∧ c ≤ b ∧ a^2 + c^2 < b^2) ∨
  (b ≤ a ∧ a ≤ c ∧ b^2 + a^2 < c^2) ∨
  (b ≤ c ∧ c ≤ a ∧ b^2 + c^2 < a^2) ∨
  (c ≤ a ∧ a ≤ b ∧ c^2 + a^2 < b^2) ∨
  (c ≤ b ∧ b ≤ a ∧ c^2 + b^2 < a^2)

theorem obtuse_triangle_x_range :
  ∀ x : ℝ, is_obtuse_triangle x (x + 1) (x + 2) → 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l516_51666


namespace NUMINAMATH_CALUDE_no_trapezoid_solution_l516_51670

theorem no_trapezoid_solution : ¬ ∃ (b₁ b₂ : ℕ), 
  (b₁ + b₂) * 40 / 2 = 1800 ∧ 
  ∃ (k : ℕ), b₁ = 2 * k + 1 ∧ 
  ∃ (m : ℕ), b₁ = 5 * m ∧
  ∃ (n : ℕ), b₂ = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_no_trapezoid_solution_l516_51670


namespace NUMINAMATH_CALUDE_fraction_evaluation_l516_51677

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l516_51677


namespace NUMINAMATH_CALUDE_vessel_base_length_l516_51615

/-- Given a cube and a rectangular vessel, proves the length of the vessel's base --/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 16) 
  (h2 : vessel_width = 15) 
  (h3 : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l516_51615


namespace NUMINAMATH_CALUDE_sophist_statements_l516_51657

/-- Represents the types of inhabitants on the Isle of Logic. -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The total number of knights on the island. -/
def num_knights : ℕ := 40

/-- The total number of liars on the island. -/
def num_liars : ℕ := 25

/-- A function that determines if a statement about the number of knights is valid for a sophist. -/
def valid_knight_statement (n : ℕ) : Prop :=
  n ≠ num_knights ∧ n = num_knights

/-- A function that determines if a statement about the number of liars is valid for a sophist. -/
def valid_liar_statement (n : ℕ) : Prop :=
  n ≠ num_liars ∧ n = num_liars + 1

/-- The main theorem stating that the only valid sophist statements are 40 knights and 26 liars. -/
theorem sophist_statements :
  (∃! k : ℕ, valid_knight_statement k) ∧
  (∃! l : ℕ, valid_liar_statement l) ∧
  valid_knight_statement 40 ∧
  valid_liar_statement 26 := by
  sorry

end NUMINAMATH_CALUDE_sophist_statements_l516_51657


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l516_51645

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_geometric : IsGeometricSequence a)
    (h_sum1 : a 1 + a 2 = 4)
    (h_sum2 : a 2 + a 3 = 12) :
    a 5 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l516_51645


namespace NUMINAMATH_CALUDE_some_number_value_l516_51637

theorem some_number_value (n : ℝ) : 9 / (1 + n / 0.5) = 1 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l516_51637


namespace NUMINAMATH_CALUDE_prime_odd_sum_l516_51684

theorem prime_odd_sum (x y : ℕ) 
  (hx : Nat.Prime x) 
  (hy : Odd y) 
  (heq : x^2 + y = 2005) : 
  x + y = 2003 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l516_51684


namespace NUMINAMATH_CALUDE_consecutive_points_length_l516_51642

/-- Given 6 consecutive points on a straight line, prove that af = 25 -/
theorem consecutive_points_length (a b c d e f : ℝ) : 
  (c - b) = 3 * (d - c) →
  (e - d) = 8 →
  (b - a) = 5 →
  (c - a) = 11 →
  (f - e) = 4 →
  (f - a) = 25 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l516_51642


namespace NUMINAMATH_CALUDE_twelve_roll_prob_l516_51689

/-- Probability of a specific outcome on a standard six-sided die -/
def die_prob : ℚ := 1 / 6

/-- Probability of rolling any number except the previous one -/
def diff_prob : ℚ := 5 / 6

/-- Number of rolls before the 8th roll -/
def pre_8th_rolls : ℕ := 6

/-- Number of rolls between 8th and 12th (exclusive) -/
def post_8th_rolls : ℕ := 3

/-- The probability that the 12th roll is the last roll, given the 8th roll is a 4 -/
theorem twelve_roll_prob : 
  (1 : ℚ) * diff_prob ^ pre_8th_rolls * die_prob * diff_prob ^ post_8th_rolls * die_prob = 5^9 / 6^11 :=
by sorry

end NUMINAMATH_CALUDE_twelve_roll_prob_l516_51689
