import Mathlib

namespace sequence_square_l2040_204052

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end sequence_square_l2040_204052


namespace largest_prime_divisor_of_factorial_sum_l2040_204078

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by
  sorry

end largest_prime_divisor_of_factorial_sum_l2040_204078


namespace f_inequality_solutions_l2040_204076

/-- The function f(x) = (x-a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x - 2)

theorem f_inequality_solutions :
  (∀ x, f 1 x > 0 ↔ x ∈ Set.Ioi 2 ∪ Set.Iic 1) ∧
  (∀ x, f 2 x < 0 → False) ∧
  (∀ a, a > 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo 2 a) ∧
  (∀ a, a < 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo a 2) :=
by sorry

end f_inequality_solutions_l2040_204076


namespace largest_five_digit_divisible_by_3_and_4_l2040_204042

theorem largest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≤ 99999) ∧ 
  (n ≥ 10000) ∧ 
  (n % 3 = 0) ∧ 
  (n % 4 = 0) ∧ 
  (∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≤ n) ∧
  n = 99996 :=
by
  sorry

end largest_five_digit_divisible_by_3_and_4_l2040_204042


namespace students_getting_B_l2040_204049

theorem students_getting_B (grade_A : ℚ) (grade_C : ℚ) (grade_D : ℚ) (grade_F : ℚ) (passing_grade : ℚ) :
  grade_A = 1/4 →
  grade_C = 1/8 →
  grade_D = 1/12 →
  grade_F = 1/24 →
  passing_grade = 0.875 →
  grade_A + grade_C + grade_D + grade_F + 3/8 = passing_grade :=
by sorry

end students_getting_B_l2040_204049


namespace equal_side_sums_exist_l2040_204045

def triangle_numbers : List ℕ := List.range 9 |>.map (· + 2016)

structure TriangleArrangement where
  positions : Fin 9 → ℕ
  is_valid : ∀ n, positions n ∈ triangle_numbers

def side_sum (arr : TriangleArrangement) (side : Fin 3) : ℕ :=
  match side with
  | 0 => arr.positions 0 + arr.positions 1 + arr.positions 2
  | 1 => arr.positions 2 + arr.positions 3 + arr.positions 4
  | 2 => arr.positions 4 + arr.positions 5 + arr.positions 0

theorem equal_side_sums_exist : 
  ∃ (arr : TriangleArrangement), ∀ (i j : Fin 3), side_sum arr i = side_sum arr j :=
sorry

end equal_side_sums_exist_l2040_204045


namespace express_1997_using_fours_l2040_204025

theorem express_1997_using_fours : 
  4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997 :=
by sorry

end express_1997_using_fours_l2040_204025


namespace company_workers_l2040_204044

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = (total / 3 : ℕ) →
  (2 * total / 10 : ℚ) * (total / 3 : ℚ) = ((2 * total / 10 : ℕ) * (total / 3 : ℕ) : ℚ) →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) = ((4 * total / 10 : ℕ) * (2 * total / 3 : ℕ) : ℚ) →
  men = 112 →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) + (total / 3 : ℚ) - (2 * total / 10 : ℚ) * (total / 3 : ℚ) = men →
  total - men = 98 :=
by sorry

end company_workers_l2040_204044


namespace cube_root_of_negative_eight_squared_l2040_204079

theorem cube_root_of_negative_eight_squared :
  ((-8^2 : ℝ) ^ (1/3 : ℝ)) = -4 := by sorry

end cube_root_of_negative_eight_squared_l2040_204079


namespace modulus_of_z_l2040_204028

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l2040_204028


namespace sqrt_equation_solution_l2040_204018

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 12) = Real.sqrt 2 / 2) → x ≥ -3/2 := by
  sorry

end sqrt_equation_solution_l2040_204018


namespace ball_bounce_distance_l2040_204002

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range (bounces - 1)) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 278.52 := by
  sorry

end ball_bounce_distance_l2040_204002


namespace sector_area_l2040_204016

theorem sector_area (θ r : ℝ) (h1 : θ = 3) (h2 : r = 4) : 
  (1/2) * θ * r^2 = 24 :=
by sorry

end sector_area_l2040_204016


namespace largest_mersenne_prime_under_500_l2040_204032

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = 2^p - 1 ∧ is_prime n

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end largest_mersenne_prime_under_500_l2040_204032


namespace candy_division_l2040_204017

theorem candy_division (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) 
  (h1 : total_candy = 344) 
  (h2 : num_students = 43) 
  (h3 : candy_per_student = total_candy / num_students) :
  candy_per_student = 8 := by
  sorry

end candy_division_l2040_204017


namespace max_non_managers_proof_l2040_204094

structure Department where
  name : String
  managers : ℕ
  ratio_managers : ℕ
  ratio_non_managers : ℕ
  active_projects : ℕ

def calculate_non_managers (d : Department) : ℕ :=
  d.managers * d.ratio_non_managers / d.ratio_managers +
  (d.active_projects + 2) / 3 +
  2

def total_non_managers (departments : List Department) : ℕ :=
  departments.foldl (fun acc d => acc + calculate_non_managers d) 0

theorem max_non_managers_proof :
  let departments : List Department := [
    { name := "Marketing", managers := 9, ratio_managers := 9, ratio_non_managers := 38, active_projects := 6 },
    { name := "HR", managers := 5, ratio_managers := 5, ratio_non_managers := 23, active_projects := 4 },
    { name := "Finance", managers := 6, ratio_managers := 6, ratio_non_managers := 31, active_projects := 5 }
  ]
  total_non_managers departments = 104 :=
by sorry

end max_non_managers_proof_l2040_204094


namespace intersection_A_complement_B_A_necessary_not_sufficient_for_B_l2040_204095

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_complement_B : 
  (A ∩ (Set.univ \ B 2)) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem A_necessary_not_sufficient_for_B :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ 0 < m ∧ m < 3 := by sorry

end intersection_A_complement_B_A_necessary_not_sufficient_for_B_l2040_204095


namespace fraction_multiplication_l2040_204008

theorem fraction_multiplication : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end fraction_multiplication_l2040_204008


namespace interest_rate_difference_l2040_204029

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) :
  principal = 6000 →
  time = 3 →
  interest_diff = 360 →
  (principal * rate2 * time) / 100 = (principal * rate1 * time) / 100 + interest_diff →
  rate2 - rate1 = 2 := by
sorry

end interest_rate_difference_l2040_204029


namespace probability_two_girls_chosen_l2040_204063

-- Define the total number of members
def total_members : ℕ := 12

-- Define the number of girls
def num_girls : ℕ := 6

-- Define the number of boys
def num_boys : ℕ := 6

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_two_girls_chosen :
  (combination num_girls 2 : ℚ) / (combination total_members 2) = 5 / 22 := by
  sorry

end probability_two_girls_chosen_l2040_204063


namespace bridge_length_l2040_204057

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed = 20 →
  crossing_time = 12.099 →
  (train_speed * crossing_time) - train_length = 131.98 := by
  sorry

end bridge_length_l2040_204057


namespace initial_concentration_proof_l2040_204039

/-- Proves that the initial concentration of a solution is 45% given the specified conditions -/
theorem initial_concentration_proof (initial_concentration : ℝ) : 
  (0.5 * initial_concentration + 0.5 * 0.25 = 0.35) → initial_concentration = 0.45 := by
  sorry

end initial_concentration_proof_l2040_204039


namespace arithmetic_sequence_property_l2040_204055

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  length : ℕ
  mean : ℚ
  first : ℚ
  last : ℚ

/-- The new mean after removing the first and last numbers -/
def new_mean (seq : ArithmeticSequence) : ℚ :=
  ((seq.length : ℚ) * seq.mean - seq.first - seq.last) / ((seq.length : ℚ) - 2)

/-- Theorem stating the property of the specific arithmetic sequence -/
theorem arithmetic_sequence_property :
  let seq := ArithmeticSequence.mk 60 42 30 70
  new_mean seq = 41.7241 := by
  sorry

end arithmetic_sequence_property_l2040_204055


namespace range_of_y_given_inequality_l2040_204038

/-- Custom multiplication operation on real numbers -/
def custom_mult (x y : ℝ) : ℝ := x * (1 - y)

/-- The theorem stating the range of y given the conditions -/
theorem range_of_y_given_inequality :
  (∀ x : ℝ, custom_mult (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ ∀ y : ℝ, a < y ∧ y < b :=
by sorry

end range_of_y_given_inequality_l2040_204038


namespace dave_deleted_eleven_apps_l2040_204066

/-- The number of apps Dave deleted -/
def apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : ℕ :=
  initial_apps - remaining_apps

/-- Theorem stating that Dave deleted 11 apps -/
theorem dave_deleted_eleven_apps : apps_deleted 16 5 = 11 := by
  sorry

end dave_deleted_eleven_apps_l2040_204066


namespace largest_prime_factor_of_9999_l2040_204034

theorem largest_prime_factor_of_9999 : ∃ (p : ℕ), p.Prime ∧ p ∣ 9999 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9999 → q ≤ p :=
sorry

end largest_prime_factor_of_9999_l2040_204034


namespace seven_valid_triples_l2040_204003

/-- The number of valid triples (a, b, c) for the prism cutting problem -/
def count_valid_triples : ℕ :=
  let b := 2023
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let c := p.2
    a ≤ b ∧ b ≤ c ∧ a * c = b * b
  ) (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card

/-- The main theorem stating there are exactly 7 valid triples -/
theorem seven_valid_triples : count_valid_triples = 7 := by
  sorry


end seven_valid_triples_l2040_204003


namespace smallest_vector_norm_l2040_204081

/-- Given a vector v such that ||v + (4, 2)|| = 10, 
    the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_vector_norm (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖u‖ ≥ ‖w‖ := by
  sorry


end smallest_vector_norm_l2040_204081


namespace sam_carrots_l2040_204088

/-- Given that Sandy grew 6 carrots and the total number of carrots grown is 9,
    prove that Sam grew 3 carrots. -/
theorem sam_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (sam_carrots : ℕ) :
  sandy_carrots = 6 → total_carrots = 9 → sam_carrots = total_carrots - sandy_carrots →
  sam_carrots = 3 := by
  sorry

#check sam_carrots

end sam_carrots_l2040_204088


namespace competition_necessarily_laughable_l2040_204092

/-- Represents the number of questions in the math competition -/
def num_questions : ℕ := 10

/-- Represents the threshold for laughable performance -/
def laughable_threshold : ℕ := 57

/-- Represents the minimum number of students for which the performance is necessarily laughable -/
def min_laughable_students : ℕ := 253

/-- Represents a student's performance on the math competition -/
structure StudentPerformance where
  correct_answers : Finset (Fin num_questions)

/-- Represents the collective performance of students in the math competition -/
def Competition (n : ℕ) := Fin n → StudentPerformance

/-- Defines when a competition performance is laughable -/
def is_laughable (comp : Competition n) : Prop :=
  ∃ (i j : Fin num_questions), i ≠ j ∧
    (∃ (students : Finset (Fin n)), students.card = laughable_threshold ∧
      (∀ s ∈ students, (i ∈ (comp s).correct_answers ∧ j ∈ (comp s).correct_answers) ∨
                       (i ∉ (comp s).correct_answers ∧ j ∉ (comp s).correct_answers)))

/-- The main theorem: any competition with at least min_laughable_students is necessarily laughable -/
theorem competition_necessarily_laughable (n : ℕ) (h : n ≥ min_laughable_students) :
  ∀ (comp : Competition n), is_laughable comp :=
sorry

end competition_necessarily_laughable_l2040_204092


namespace quadratic_inequality_solution_l2040_204027

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - x - 4 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 4/3) := by
  sorry

end quadratic_inequality_solution_l2040_204027


namespace tangent_line_proof_l2040_204086

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 5*x^2 - 5

-- Define the given line
def l₁ (z y : ℝ) : Prop := 2*z - 6*y + 1 = 0

-- Define the tangent line
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) lies on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    l₂ x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 10*x₀ = -3) ∧
    -- The two lines are perpendicular
    ∀ (z₁ y₁ z₂ y₂ : ℝ),
      l₁ z₁ y₁ ∧ l₁ z₂ y₂ ∧ z₁ ≠ z₂ →
      (y₁ - y₂) / (z₁ - z₂) * (-1/3) = -1 :=
by sorry

end tangent_line_proof_l2040_204086


namespace gabby_needs_ten_more_dollars_l2040_204009

def makeup_set_cost : ℕ := 65
def gabby_initial_savings : ℕ := 35
def mom_additional_money : ℕ := 20

theorem gabby_needs_ten_more_dollars : 
  makeup_set_cost - (gabby_initial_savings + mom_additional_money) = 10 := by
sorry

end gabby_needs_ten_more_dollars_l2040_204009


namespace average_pushups_l2040_204033

theorem average_pushups (david zachary emily : ℕ) : 
  david = 510 ∧ 
  david = zachary + 210 ∧ 
  david = emily + 132 → 
  (david + zachary + emily) / 3 = 396 := by
    sorry

end average_pushups_l2040_204033


namespace inequality_always_true_iff_a_in_range_l2040_204097

theorem inequality_always_true_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
by sorry

end inequality_always_true_iff_a_in_range_l2040_204097


namespace tyler_meal_choices_l2040_204074

-- Define the number of options for each category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4
def num_drinks : ℕ := 3

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 3

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats) * (Nat.choose num_vegetables vegetables_to_choose) * (num_desserts) * (num_drinks) = 360 := by
  sorry


end tyler_meal_choices_l2040_204074


namespace burger_cost_l2040_204065

theorem burger_cost (burger soda : ℕ) 
  (alice_purchase : 4 * burger + 3 * soda = 440)
  (bob_purchase : 3 * burger + 2 * soda = 330) :
  burger = 110 := by
sorry

end burger_cost_l2040_204065


namespace intersection_equality_l2040_204062

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem intersection_equality (m : ℝ) : B ∩ A m = B → m = 4 := by
  sorry

end intersection_equality_l2040_204062


namespace rectangle_circle_area_ratio_l2040_204020

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r ^ 2) = 18 / Real.pi ^ 2 := by
  sorry

end rectangle_circle_area_ratio_l2040_204020


namespace f_inequality_l2040_204071

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
variable (h_deriv : ∀ x, HasDerivAt f (f' x) x)

-- Define the condition that f(x) > f'(x) for all x
variable (h_cond : ∀ x, f x > f' x)

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end f_inequality_l2040_204071


namespace cube_with_holes_surface_area_l2040_204060

/-- Represents a cube with holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let area_removed_by_holes := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 6 * cube.hole_side_length^2
  original_surface_area - area_removed_by_holes + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and hole side length 2 has total surface area 270 -/
theorem cube_with_holes_surface_area :
  total_surface_area { edge_length := 5, hole_side_length := 2 } = 270 := by
  sorry

end cube_with_holes_surface_area_l2040_204060


namespace sin_graph_shift_l2040_204035

open Real

theorem sin_graph_shift (f g : ℝ → ℝ) (ω : ℝ) (h : ω = 2) :
  (∀ x, f x = sin (ω * x + π / 6)) →
  (∀ x, g x = sin (ω * x)) →
  ∃ shift, ∀ x, f x = g (x - shift) ∧ shift = π / 12 :=
sorry

end sin_graph_shift_l2040_204035


namespace exactly_two_solutions_l2040_204089

/-- The number of solutions to the system of equations -/
def num_solutions : ℕ := 2

/-- A solution to the system of equations is a triple of positive integers (x, y, z) -/
def is_solution (x y z : ℕ+) : Prop :=
  x * y + x * z = 255 ∧ x * z - y * z = 224

/-- The theorem stating that there are exactly two solutions -/
theorem exactly_two_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = num_solutions ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ is_solution x y z) :=
sorry

end exactly_two_solutions_l2040_204089


namespace cereal_original_price_l2040_204041

def initial_money : ℝ := 60
def celery_price : ℝ := 5
def bread_price : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_price : ℝ := 1
def potato_quantity : ℕ := 6
def money_left : ℝ := 26
def cereal_discount : ℝ := 0.5

theorem cereal_original_price :
  let milk_price := milk_original_price * (1 - milk_discount)
  let potato_total := potato_price * potato_quantity
  let spent_on_known_items := celery_price + bread_price + milk_price + potato_total
  let total_spent := initial_money - money_left
  let cereal_discounted_price := total_spent - spent_on_known_items
  cereal_discounted_price / (1 - cereal_discount) = 12 := by sorry

end cereal_original_price_l2040_204041


namespace juice_box_days_l2040_204098

theorem juice_box_days (num_children : ℕ) (school_weeks : ℕ) (total_juice_boxes : ℕ) :
  num_children = 3 →
  school_weeks = 25 →
  total_juice_boxes = 375 →
  (total_juice_boxes / (num_children * school_weeks) : ℚ) = 5 := by
  sorry

end juice_box_days_l2040_204098


namespace sequence_sixth_term_l2040_204019

theorem sequence_sixth_term :
  let seq : ℕ → ℚ := fun n => 1 / (n * (n + 1))
  seq 6 = 1 / 42 := by
  sorry

end sequence_sixth_term_l2040_204019


namespace rectangle_side_length_l2040_204006

/-- Given three rectangles with equal areas and integer sides, where one side is 29, prove that another side is 870 -/
theorem rectangle_side_length (a b k l : ℕ) : 
  let S := 29 * (a + b)
  a * k = S ∧ 
  b * l = S ∧ 
  k * l = 29 * (k + l) →
  k = 870 := by
sorry

end rectangle_side_length_l2040_204006


namespace price_large_bottle_correct_l2040_204053

/-- The price of a large bottle, given the following conditions:
  * 1365 large bottles were purchased at this price
  * 720 small bottles were purchased at $1.42 each
  * The average price of all bottles was approximately $1.73
-/
def price_large_bottle : ℝ := 1.89

theorem price_large_bottle_correct : 
  let num_large : ℕ := 1365
  let num_small : ℕ := 720
  let price_small : ℝ := 1.42
  let avg_price : ℝ := 1.73
  let total_bottles : ℕ := num_large + num_small
  let total_cost : ℝ := (num_large : ℝ) * price_large_bottle + (num_small : ℝ) * price_small
  abs (total_cost / (total_bottles : ℝ) - avg_price) < 0.01 ∧ 
  abs (price_large_bottle - 1.89) < 0.01 := by
sorry

end price_large_bottle_correct_l2040_204053


namespace water_depth_in_cistern_l2040_204004

/-- Calculates the depth of water in a rectangular cistern given its dimensions and wet surface area. -/
theorem water_depth_in_cistern
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h1 : length = 8)
  (h2 : width = 6)
  (h3 : total_wet_surface_area = 83) :
  ∃ (depth : ℝ), depth = 1.25 ∧ 
    total_wet_surface_area = length * width + 2 * (length + width) * depth :=
by sorry


end water_depth_in_cistern_l2040_204004


namespace alternating_walk_forms_cycle_l2040_204073

/-- Represents a direction of turn -/
inductive Direction
| Left
| Right

/-- Represents the island as a graph -/
structure Island where
  -- The set of vertices (junctions)
  vertices : Type
  -- The edges (roads) between vertices
  edges : vertices → vertices → Prop
  -- Every vertex has exactly three edges
  three_roads : ∀ v : vertices, ∃! (n : Nat), n = 3 ∧ (∃ (adjacent : Finset vertices), adjacent.card = n ∧ ∀ u ∈ adjacent, edges v u)

/-- Represents a walk on the island -/
def Walk (island : Island) : Type :=
  Nat → island.vertices × Direction

/-- A walk is alternating if it alternates between left and right turns -/
def IsAlternating (walk : Walk island) : Prop :=
  ∀ n : Nat, 
    (walk n).2 ≠ (walk (n + 1)).2

/-- The main theorem: any alternating walk on a finite island will eventually form a cycle -/
theorem alternating_walk_forms_cycle (island : Island) (walk : Walk island) 
    (finite_island : Finite island.vertices) (alternating : IsAlternating walk) : 
    ∃ (start finish : Nat), start < finish ∧ (walk start).1 = (walk finish).1 := by
  sorry


end alternating_walk_forms_cycle_l2040_204073


namespace platform_length_l2040_204082

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmph = 75 →
  crossing_time = 24 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    platform_length = (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end platform_length_l2040_204082


namespace comic_books_total_l2040_204096

theorem comic_books_total (jake_books : ℕ) (brother_difference : ℕ) : 
  jake_books = 36 → brother_difference = 15 → 
  jake_books + (jake_books + brother_difference) = 87 := by
  sorry

end comic_books_total_l2040_204096


namespace apple_theorem_l2040_204068

def apple_problem (initial_apples : ℕ) : ℕ :=
  let after_jill := initial_apples - (initial_apples * 30 / 100)
  let after_june := after_jill - (after_jill * 20 / 100)
  let after_friend := after_june - 2
  after_friend - (after_friend * 10 / 100)

theorem apple_theorem :
  apple_problem 150 = 74 := by sorry

end apple_theorem_l2040_204068


namespace equivalence_point_cost_effectiveness_l2040_204080

-- Define the full ticket price
def full_price : ℝ := 240

-- Define the charge functions for Travel Agency A and B
def charge_A (x : ℝ) : ℝ := 120 * x + 240
def charge_B (x : ℝ) : ℝ := 144 * x + 144

-- Theorem for the equivalence point
theorem equivalence_point :
  ∃ x : ℝ, charge_A x = charge_B x ∧ x = 4 := by sorry

-- Theorem for cost-effectiveness comparison
theorem cost_effectiveness (x : ℝ) :
  (x < 4 → charge_B x < charge_A x) ∧
  (x > 4 → charge_A x < charge_B x) := by sorry

end equivalence_point_cost_effectiveness_l2040_204080


namespace may_greatest_drop_l2040_204001

/-- Represents the months in the first half of 2022 -/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june

/-- Represents the price change for each month -/
def priceChange (m : Month) : ℝ :=
  match m with
  | .january => -1.0
  | .february => 1.5
  | .march => -3.0
  | .april => 2.0
  | .may => -4.0
  | .june => -1.5

/-- The economic event occurred in May -/
def economicEventMonth : Month := .may

/-- Defines the greatest monthly drop in price -/
def hasGreatestDrop (m : Month) : Prop :=
  ∀ m', priceChange m ≤ priceChange m'

/-- Theorem stating that May has the greatest monthly drop in price -/
theorem may_greatest_drop :
  hasGreatestDrop .may :=
sorry

end may_greatest_drop_l2040_204001


namespace battery_charging_time_l2040_204084

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_rate : ℝ  -- Percentage charged per hour
  initial_time : ℝ  -- Time for initial charge in minutes
  additional_time : ℝ  -- Additional time to reach certain percentage in minutes

/-- Calculates the total charging time for a mobile battery -/
def total_charging_time (b : BatteryCharging) : ℝ :=
  b.initial_time + b.additional_time

/-- Theorem: The total charging time for the given battery is 255 minutes -/
theorem battery_charging_time :
  let b : BatteryCharging := {
    initial_rate := 20,
    initial_time := 60,
    additional_time := 195
  }
  total_charging_time b = 255 := by
  sorry

end battery_charging_time_l2040_204084


namespace isosceles_triangle_perimeter_l2040_204007

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 := by  -- Perimeter is 17
  sorry

end isosceles_triangle_perimeter_l2040_204007


namespace relationship_abc_l2040_204067

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 2 → b = 2^(0.8 : ℝ) → c = 2 * Real.log 2 / Real.log 5 → c < a ∧ a < b := by
  sorry

end relationship_abc_l2040_204067


namespace system_solution_l2040_204085

theorem system_solution (a b : ℚ) : 
  (∃ b, 2 * 1 - b * 2 = 1) →
  (∃ a, a * 1 + 1 = 2) →
  (∃! x y : ℚ, a * x + y = 2 ∧ 2 * x - b * y = 1 ∧ x = 4/5 ∧ y = 6/5) :=
by sorry

end system_solution_l2040_204085


namespace defective_books_relative_frequency_l2040_204059

/-- The relative frequency of an event is the ratio of the number of times 
    the event occurs to the total number of trials or experiments. -/
def relative_frequency (event_occurrences : ℕ) (total_trials : ℕ) : ℚ :=
  event_occurrences / total_trials

/-- Given a batch of 100 randomly selected books with 5 defective books,
    prove that the relative frequency of defective books is 0.05. -/
theorem defective_books_relative_frequency :
  let total_books : ℕ := 100
  let defective_books : ℕ := 5
  relative_frequency defective_books total_books = 5 / 100 := by
  sorry

#eval (5 : ℚ) / 100  -- To verify the result is indeed 0.05

end defective_books_relative_frequency_l2040_204059


namespace circle_area_from_circumference_l2040_204026

theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 30 * π → π * r^2 = 225 * π :=
by
  sorry

end circle_area_from_circumference_l2040_204026


namespace sector_arc_length_l2040_204036

/-- The length of an arc of a sector with given central angle and radius -/
def arcLength (centralAngle : Real) (radius : Real) : Real :=
  radius * centralAngle

theorem sector_arc_length :
  let centralAngle : Real := π / 5
  let radius : Real := 20
  arcLength centralAngle radius = 4 * π := by sorry

end sector_arc_length_l2040_204036


namespace volleyball_game_employees_l2040_204014

theorem volleyball_game_employees (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) :
  managers = 3 →
  teams = 3 →
  people_per_team = 2 →
  teams * people_per_team - managers = 3 :=
by
  sorry

end volleyball_game_employees_l2040_204014


namespace singh_gain_l2040_204046

/-- Represents the game outcome for three players -/
structure GameOutcome where
  ashtikar : ℚ
  singh : ℚ
  bhatia : ℚ

/-- The initial amount each player starts with -/
def initial_amount : ℚ := 70

/-- The theorem stating Singh's gain in the game -/
theorem singh_gain (outcome : GameOutcome) : 
  outcome.ashtikar + outcome.singh + outcome.bhatia = 3 * initial_amount ∧
  outcome.ashtikar = (1/2) * outcome.singh ∧
  outcome.bhatia = (1/4) * outcome.singh →
  outcome.singh - initial_amount = 50 := by
sorry

end singh_gain_l2040_204046


namespace paula_and_karl_ages_sum_l2040_204093

theorem paula_and_karl_ages_sum (P K : ℕ) : 
  (P - 5 = 3 * (K - 5)) →  -- 5 years ago, Paula was 3 times as old as Karl
  (P + 6 = 2 * (K + 6)) →  -- In 6 years, Paula will be twice as old as Karl
  P + K = 54 :=            -- The sum of their current ages is 54
by sorry

end paula_and_karl_ages_sum_l2040_204093


namespace existence_of_special_integers_l2040_204083

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- nonzero integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- pairwise distinct
  a + b + c = 0 ∧           -- sum is zero
  ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end existence_of_special_integers_l2040_204083


namespace square_plus_cube_equals_one_l2040_204050

theorem square_plus_cube_equals_one : 3^2 + (-2)^3 = 1 := by
  sorry

end square_plus_cube_equals_one_l2040_204050


namespace modulus_of_z_l2040_204011

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (1 + I) = I) : abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l2040_204011


namespace central_angle_for_given_arc_central_angle_proof_l2040_204064

/-- Given a circle with radius 100mm and an arc length of 300mm,
    the central angle corresponding to this arc is 3 radians. -/
theorem central_angle_for_given_arc : ℝ → ℝ → ℝ → Prop :=
  λ radius arc_length angle =>
    radius = 100 ∧ arc_length = 300 → angle = 3

/-- The theorem proof -/
theorem central_angle_proof :
  ∃ (angle : ℝ), central_angle_for_given_arc 100 300 angle :=
by
  sorry

end central_angle_for_given_arc_central_angle_proof_l2040_204064


namespace polynomial_characterization_l2040_204021

variable (f g : ℝ → ℝ)

def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) ≥ t * f x + (1 - t) * f y

theorem polynomial_characterization
  (hf_concave : IsConcave f)
  (hg_continuous : Continuous g)
  (h_equality : ∀ x y : ℝ, f (x + y) + f (x - y) - 2 * f x = g x * y^2) :
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x + B * x^2 + C :=
sorry

end polynomial_characterization_l2040_204021


namespace triangle_area_l2040_204058

/-- The area of a triangle with sides 10, 24, and 26 is 120 square units -/
theorem triangle_area : ∀ (a b c : ℝ),
  a = 10 ∧ b = 24 ∧ c = 26 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
   Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 120) :=
by sorry

end triangle_area_l2040_204058


namespace count_equal_f_is_501_l2040_204087

/-- f(n) denotes the number of 1's in the base-2 representation of n -/
def f (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 1 and 2002 (inclusive) where f(n) = f(n+1) -/
def count_equal_f : ℕ := sorry

theorem count_equal_f_is_501 : count_equal_f = 501 := by sorry

end count_equal_f_is_501_l2040_204087


namespace max_value_sqrt_sum_l2040_204013

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  ∃ (M : ℝ), M = 3 * Real.sqrt 8 ∧ 
  Real.sqrt (3*x + 2) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 2) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 6 ∧
    Real.sqrt (3*x' + 2) + Real.sqrt (3*y' + 2) + Real.sqrt (3*z' + 2) = M :=
by sorry

end max_value_sqrt_sum_l2040_204013


namespace f_equation_l2040_204047

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_equation : ∀ x : ℝ, f (x + 1) = x^2 - 5*x + 4 → f x = x^2 - 7*x + 10 := by
  sorry

end f_equation_l2040_204047


namespace ceasar_pages_read_l2040_204075

/-- The number of pages Ceasar has already read -/
def pages_read (total_pages remaining_pages : ℕ) : ℕ :=
  total_pages - remaining_pages

theorem ceasar_pages_read :
  pages_read 563 416 = 147 := by
  sorry

end ceasar_pages_read_l2040_204075


namespace bill_face_value_l2040_204056

/-- Calculates the face value of a bill given the true discount, interest rate, and time period. -/
def calculate_face_value (true_discount : ℚ) (interest_rate : ℚ) (time_months : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Proves that the face value of the bill is 1575 given the specified conditions. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let interest_rate : ℚ := 16
  let time_months : ℚ := 9
  calculate_face_value true_discount interest_rate time_months = 1575 := by
  sorry

#eval calculate_face_value 189 16 9

end bill_face_value_l2040_204056


namespace parallel_line_y_intercept_l2040_204023

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

theorem parallel_line_y_intercept (b : Line) :
  parallel b { slope := -3, yIntercept := 6 } →
  pointOnLine b 4 (-1) →
  b.yIntercept = 11 := by sorry

end parallel_line_y_intercept_l2040_204023


namespace linear_coefficient_of_quadratic_l2040_204015

/-- The coefficient of the linear term in the quadratic equation x^2 - x = 0 is -1 -/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (fun x => x^2 - x) = (fun x => x^2 - 1*x) :=
by sorry

end linear_coefficient_of_quadratic_l2040_204015


namespace money_problem_l2040_204022

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a - b > 32)
  (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := by
  sorry

end money_problem_l2040_204022


namespace stratified_sampling_theorem_l2040_204043

theorem stratified_sampling_theorem (total_population : ℕ) (female_population : ℕ) (sample_size : ℕ) (female_sample : ℕ) :
  total_population = 2400 →
  female_population = 1000 →
  female_sample = 40 →
  (female_sample : ℚ) / sample_size = (female_population : ℚ) / total_population →
  sample_size = 96 :=
by
  sorry

end stratified_sampling_theorem_l2040_204043


namespace no_solution_implies_a_leq_two_l2040_204051

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end no_solution_implies_a_leq_two_l2040_204051


namespace kantana_chocolates_l2040_204054

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := sorry

/-- Represents the number of Saturdays in the month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates bought for Charlie's birthday -/
def chocolates_for_charlie : ℕ := 10

/-- Represents the total number of chocolates bought in the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana buys 2 chocolates for herself each Saturday -/
theorem kantana_chocolates : 
  chocolates_for_self = 2 ∧ 
  (chocolates_for_self + 1) * saturdays_in_month + chocolates_for_charlie = total_chocolates :=
by sorry

end kantana_chocolates_l2040_204054


namespace angle_with_complement_40percent_of_supplement_is_30_degrees_l2040_204061

theorem angle_with_complement_40percent_of_supplement_is_30_degrees :
  ∀ x : ℝ,
  (x > 0) →
  (x < 90) →
  (90 - x = (2/5) * (180 - x)) →
  x = 30 :=
by
  sorry

end angle_with_complement_40percent_of_supplement_is_30_degrees_l2040_204061


namespace solution_value_l2040_204070

theorem solution_value (m n : ℝ) : 
  (∀ x, x^2 - m*x + n ≤ 0 ↔ -5 ≤ x ∧ x ≤ 1) →
  ((-5)^2 - m*(-5) + n = 0) →
  (1^2 - m*1 + n = 0) →
  m - n = 1 :=
by
  sorry

end solution_value_l2040_204070


namespace max_value_fraction_l2040_204091

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/2 := by
sorry

end max_value_fraction_l2040_204091


namespace f_6n_l2040_204048

def f : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    if n % 6 = 0 ∨ n % 6 = 1 then f n + 3
    else if n % 6 = 2 ∨ n % 6 = 5 then f n + 1
    else f n + 2

theorem f_6n (n : ℕ) : f (6 * n) = 12 * n := by
  sorry

end f_6n_l2040_204048


namespace discount_equation_l2040_204099

theorem discount_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 164)
  (h3 : final_price = original_price * (1 - x)^2) :
  200 * (1 - x)^2 = 164 := by
  sorry

end discount_equation_l2040_204099


namespace alcohol_percentage_solution_y_l2040_204040

theorem alcohol_percentage_solution_y :
  let alcohol_x : ℝ := 0.1  -- 10% alcohol in solution x
  let volume_x : ℝ := 300   -- 300 mL of solution x
  let volume_y : ℝ := 900   -- 900 mL of solution y
  let total_volume : ℝ := volume_x + volume_y
  let final_alcohol_percentage : ℝ := 0.25  -- 25% alcohol in final solution
  let alcohol_y : ℝ := (final_alcohol_percentage * total_volume - alcohol_x * volume_x) / volume_y
  alcohol_y = 0.3  -- 30% alcohol in solution y
  := by sorry

end alcohol_percentage_solution_y_l2040_204040


namespace work_completion_time_l2040_204031

def work_rate (days : ℕ) : ℚ := 1 / days

def johnson_rate : ℚ := work_rate 10
def vincent_rate : ℚ := work_rate 40
def alice_rate : ℚ := work_rate 20
def bob_rate : ℚ := work_rate 30

def day1_rate : ℚ := johnson_rate + vincent_rate
def day2_rate : ℚ := alice_rate + bob_rate

def two_day_cycle_rate : ℚ := day1_rate + day2_rate

theorem work_completion_time : ∃ n : ℕ, n * two_day_cycle_rate ≥ 1 ∧ n * 2 = 10 := by
  sorry

end work_completion_time_l2040_204031


namespace crayons_difference_l2040_204077

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between the number of crayons lost and the number of crayons given away is 322. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
    (h1 : initial = 110)
    (h2 : given_away = 90)
    (h3 : lost = 412) :
    lost - given_away = 322 := by
  sorry

end crayons_difference_l2040_204077


namespace arccos_neg_half_equals_two_pi_thirds_l2040_204012

theorem arccos_neg_half_equals_two_pi_thirds :
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end arccos_neg_half_equals_two_pi_thirds_l2040_204012


namespace sqrt_meaningful_range_l2040_204005

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end sqrt_meaningful_range_l2040_204005


namespace audrey_lost_six_pieces_l2040_204000

/-- Represents the number of pieces in a chess game -/
structure ChessGame where
  total_pieces : ℕ
  audrey_pieces : ℕ
  thomas_pieces : ℕ

/-- The initial state of a chess game -/
def initial_chess_game : ChessGame :=
  { total_pieces := 32
  , audrey_pieces := 16
  , thomas_pieces := 16 }

/-- The final state of the chess game after pieces are lost -/
def final_chess_game : ChessGame :=
  { total_pieces := 21
  , audrey_pieces := 21 - (initial_chess_game.thomas_pieces - 5)
  , thomas_pieces := initial_chess_game.thomas_pieces - 5 }

/-- Theorem stating that Audrey lost 6 pieces -/
theorem audrey_lost_six_pieces :
  initial_chess_game.audrey_pieces - final_chess_game.audrey_pieces = 6 := by
  sorry


end audrey_lost_six_pieces_l2040_204000


namespace least_common_period_is_36_l2040_204030

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def LeastCommonPeriod : ℝ := 36

/-- Main theorem: The least common positive period for all functions satisfying the condition is 36 -/
theorem least_common_period_is_36 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
  (∀ p : ℝ, p > 0 → IsPeriodic f p → p ≥ LeastCommonPeriod) ∧
  (∃ f : ℝ → ℝ, SatisfiesCondition f ∧ IsPeriodic f LeastCommonPeriod) :=
sorry

end least_common_period_is_36_l2040_204030


namespace gcd_n_cube_plus_27_and_n_plus_3_l2040_204090

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end gcd_n_cube_plus_27_and_n_plus_3_l2040_204090


namespace sqrt_meaningful_range_l2040_204072

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end sqrt_meaningful_range_l2040_204072


namespace sequence_ratio_l2040_204037

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the sum of certain terms to another term equals 5/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (1 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < 4 ∧  -- arithmetic sequence condition
  (∃ r : ℝ, r > 0 ∧ b₁ = r ∧ b₂ = r^2 ∧ b₃ = r^3 ∧ 4 = r^4) →  -- geometric sequence condition
  (a₁ + a₂) / b₂ = 5/2 := by
sorry

end sequence_ratio_l2040_204037


namespace muffin_banana_price_ratio_l2040_204024

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  (5 * muffin_price + 4 * banana_price = 20) →
  (3 * muffin_price + 18 * banana_price = 60) →
  muffin_price / banana_price = 13 / 4 := by
sorry

end muffin_banana_price_ratio_l2040_204024


namespace function_is_identity_or_reflection_l2040_204010

-- Define the function f
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- State the theorem
theorem function_is_identity_or_reflection (a b : ℝ) :
  (∀ x : ℝ, f a b (f a b x) = x) →
  ((a = 1 ∧ b = 0) ∨ ∃ c : ℝ, a = -1 ∧ ∀ x : ℝ, f a b x = -x + c) :=
by sorry

end function_is_identity_or_reflection_l2040_204010


namespace gcd_fx_x_l2040_204069

def f (x : ℤ) : ℤ := (3*x+4)*(5*x+6)*(11*x+9)*(x+7)

theorem gcd_fx_x (x : ℤ) (h : ∃ k : ℤ, x = 35622 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 378 := by
  sorry

end gcd_fx_x_l2040_204069
