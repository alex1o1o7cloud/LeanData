import Mathlib

namespace equation_solution_range_l1160_116090

theorem equation_solution_range (k : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ (x^2 + k*x + 3) / (x - 1) = 3*x + k) ↔ 
  (k = -33/8 ∨ k = -4 ∨ k ≥ -3) := by
sorry

end equation_solution_range_l1160_116090


namespace total_subjects_is_41_l1160_116039

/-- The number of subjects taken by Monica -/
def monica_subjects : ℕ := 10

/-- The number of subjects taken by Marius -/
def marius_subjects : ℕ := monica_subjects + 4

/-- The number of subjects taken by Millie -/
def millie_subjects : ℕ := marius_subjects + 3

/-- The total number of subjects taken by all three students -/
def total_subjects : ℕ := monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating that the total number of subjects is 41 -/
theorem total_subjects_is_41 : total_subjects = 41 := by
  sorry

end total_subjects_is_41_l1160_116039


namespace arrangements_starting_with_vowel_l1160_116077

def word : String := "basics"

def is_vowel (c : Char) : Bool :=
  c = 'a' || c = 'e' || c = 'i' || c = 'o' || c = 'u'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def permutations_with_repetition (total : Nat) (repeated : List Nat) : Nat :=
  factorial total / (repeated.map factorial).prod

theorem arrangements_starting_with_vowel :
  let total_letters := word.length
  let vowels := count_vowels word
  let consonants := total_letters - vowels
  let arrangements := 
    vowels * permutations_with_repetition (total_letters - 1) [consonants, vowels - 1, 1]
  arrangements = 120 := by
  sorry

end arrangements_starting_with_vowel_l1160_116077


namespace pyramid_cube_tiling_exists_l1160_116016

/-- A shape constructed from a cube with a pyramid on one face -/
structure PyramidCube where
  -- The edge length of the base cube
  cube_edge : ℝ
  -- The height of the pyramid (assumed to be equal to cube_edge)
  pyramid_height : ℝ
  -- Assumption that the pyramid height equals the cube edge length
  height_eq_edge : pyramid_height = cube_edge

/-- A tiling of 3D space using congruent copies of a shape -/
structure Tiling (shape : PyramidCube) where
  -- The set of positions (as points in ℝ³) where shapes are placed
  positions : Set (Fin 3 → ℝ)
  -- Property ensuring the tiling is seamless (no gaps)
  seamless : sorry
  -- Property ensuring the tiling has no overlaps
  no_overlap : sorry

/-- Theorem stating that a space-filling tiling exists for the PyramidCube shape -/
theorem pyramid_cube_tiling_exists :
  ∃ (shape : PyramidCube) (tiling : Tiling shape), True :=
sorry

end pyramid_cube_tiling_exists_l1160_116016


namespace binomial_150_150_l1160_116038

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end binomial_150_150_l1160_116038


namespace partition_remainder_l1160_116097

theorem partition_remainder (S : Finset ℕ) : 
  S.card = 15 → 
  (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000 = 406 := by
  sorry

#eval (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000

end partition_remainder_l1160_116097


namespace car_speed_problem_l1160_116057

theorem car_speed_problem (D : ℝ) (h : D > 0) :
  let t1 := D / 3 / 80
  let t2 := D / 3 / 30
  let t3 := D / 3 / 48
  45 = D / (t1 + t2 + t3) :=
by
  sorry

#check car_speed_problem

end car_speed_problem_l1160_116057


namespace weaving_increase_proof_l1160_116015

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial weaving output on the first day -/
def initial_output : ℚ := 5

/-- Represents the total number of days -/
def total_days : ℕ := 30

/-- Represents the total amount of fabric woven over the period -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  (initial_output + (total_days - 1) * daily_increase / 2) * total_days = total_output := by
  sorry

end weaving_increase_proof_l1160_116015


namespace comparison_theorem_l1160_116075

theorem comparison_theorem (a b c : ℝ) 
  (ha : a = Real.log 1.01)
  (hb : b = 1 / 101)
  (hc : c = Real.sin 0.01) :
  a > b ∧ c > a := by sorry

end comparison_theorem_l1160_116075


namespace cube_root_sum_of_cubes_l1160_116028

theorem cube_root_sum_of_cubes : 
  (20^3 + 70^3 + 110^3 : ℝ)^(1/3) = 120 := by sorry

end cube_root_sum_of_cubes_l1160_116028


namespace quadratic_equation_root_l1160_116005

theorem quadratic_equation_root (b : ℝ) : 
  (2 * (4 : ℝ)^2 + b * 4 - 44 = 0) → b = 3 := by
  sorry

end quadratic_equation_root_l1160_116005


namespace diana_video_game_time_l1160_116000

def video_game_time_per_hour_read : ℕ := 30
def raise_percentage : ℚ := 0.2
def chores_for_bonus_time : ℕ := 2
def bonus_time_per_chore_set : ℕ := 10
def max_bonus_time_from_chores : ℕ := 60
def hours_read : ℕ := 8
def chores_completed : ℕ := 10

theorem diana_video_game_time : 
  let base_time := hours_read * video_game_time_per_hour_read
  let raised_time := base_time + (base_time * raise_percentage).floor
  let chore_bonus_time := min (chores_completed / chores_for_bonus_time * bonus_time_per_chore_set) max_bonus_time_from_chores
  raised_time + chore_bonus_time = 338 := by
sorry

end diana_video_game_time_l1160_116000


namespace sequence_general_formula_l1160_116051

theorem sequence_general_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 3 ∧
  (∀ n : ℕ+, S n = 2 * n * a (n + 1) - 3 * n^2 - 4 * n) →
  ∀ n : ℕ+, a n = 2 * n + 1 :=
by sorry

end sequence_general_formula_l1160_116051


namespace cuboid_volume_l1160_116020

/-- A cuboid with given height and base area has the specified volume. -/
theorem cuboid_volume (height : ℝ) (base_area : ℝ) :
  height = 13 → base_area = 14 → height * base_area = 182 := by
  sorry

end cuboid_volume_l1160_116020


namespace system_solution_l1160_116067

theorem system_solution : 
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -2) ∧ 
    (5 * x + 2 * y = 8) ∧ 
    (x = 20 / 23) ∧ 
    (y = 42 / 23) := by
  sorry

end system_solution_l1160_116067


namespace f_constant_on_interval_inequality_solution_condition_l1160_116047

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem 1: f(x) is constant on the interval [-3, 1]
theorem f_constant_on_interval :
  ∀ x y : ℝ, x ∈ Set.Icc (-3) 1 → y ∈ Set.Icc (-3) 1 → f x = f y :=
sorry

-- Theorem 2: For f(x) - a ≤ 0 to have a solution, a must be ≥ 4
theorem inequality_solution_condition :
  ∀ a : ℝ, (∃ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 :=
sorry

end f_constant_on_interval_inequality_solution_condition_l1160_116047


namespace marble_count_l1160_116040

/-- The total number of marbles owned by Albert, Angela, Allison, Addison, and Alex -/
def total_marbles (allison angela albert addison alex : ℕ) : ℕ :=
  allison + angela + albert + addison + alex

/-- Theorem stating the total number of marbles given the conditions -/
theorem marble_count :
  ∀ (allison angela albert addison alex : ℕ),
    allison = 28 →
    angela = allison + 8 →
    albert = 3 * angela →
    addison = 2 * albert →
    alex = allison + 5 →
    alex = angela / 2 →
    total_marbles allison angela albert addison alex = 421 := by
  sorry


end marble_count_l1160_116040


namespace k_value_at_4_l1160_116012

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the properties of k
def k_properties (k : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, h a = 0 ∧ h b = 0 ∧ h c = 0 ∧
    ∀ x, k x = (x - a^2) * (x - b^2) * (x - c^2)) ∧
  k 0 = 1

-- Theorem statement
theorem k_value_at_4 (k : ℝ → ℝ) (hk : k_properties k) : k 4 = 15 := by
  sorry

end k_value_at_4_l1160_116012


namespace least_n_for_jumpy_l1160_116072

/-- A permutation of 2021 elements -/
def Permutation := Fin 2021 → Fin 2021

/-- A function that reorders up to 1232 elements in a permutation -/
def reorder_1232 (p : Permutation) : Permutation :=
  sorry

/-- A function that reorders up to n elements in a permutation -/
def reorder_n (n : ℕ) (p : Permutation) : Permutation :=
  sorry

/-- The identity permutation -/
def id_perm : Permutation :=
  sorry

theorem least_n_for_jumpy :
  ∀ n : ℕ,
    (∀ p : Permutation,
      ∃ q : Permutation,
        reorder_n n (reorder_1232 p) = id_perm) ↔
    n ≥ 1234 :=
  sorry

end least_n_for_jumpy_l1160_116072


namespace hoseok_workbook_days_l1160_116049

/-- The number of days Hoseok solved the workbook -/
def days_solved : ℕ := 12

/-- The number of pages Hoseok solves per day -/
def pages_per_day : ℕ := 4

/-- The total number of pages Hoseok has solved -/
def total_pages : ℕ := 48

/-- Theorem stating that the number of days Hoseok solved the workbook is correct -/
theorem hoseok_workbook_days : 
  days_solved = total_pages / pages_per_day :=
by sorry

end hoseok_workbook_days_l1160_116049


namespace intersection_at_one_point_l1160_116014

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 2*x + 2 = -2*x - 2) ↔ b = 1 := by
  sorry

end intersection_at_one_point_l1160_116014


namespace four_digit_divisible_by_3_l1160_116022

/-- A function that returns true if a four-digit number of the form 258n is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9 ∧ (2580 + n) % 3 = 0

/-- Theorem stating that a four-digit number 258n is divisible by 3 iff n is 0, 3, 6, or 9 -/
theorem four_digit_divisible_by_3 :
  ∀ n : Nat, isDivisibleBy3 n ↔ n = 0 ∨ n = 3 ∨ n = 6 ∨ n = 9 := by
  sorry

end four_digit_divisible_by_3_l1160_116022


namespace commute_days_l1160_116046

theorem commute_days (bus_to_work bus_to_home train_days train_both : ℕ) : 
  bus_to_work = 12 → 
  bus_to_home = 20 → 
  train_days = 14 → 
  train_both = 2 → 
  ∃ x : ℕ, x = 23 ∧ 
    x = (bus_to_home - bus_to_work + train_both) + 
        (bus_to_work - train_both) + 
        (train_days - (bus_to_home - bus_to_work)) + 
        train_both :=
by sorry

end commute_days_l1160_116046


namespace prob_two_or_fewer_white_eq_23_28_l1160_116076

/-- The number of white balls in the bag -/
def white_balls : Nat := 5

/-- The number of red balls in the bag -/
def red_balls : Nat := 3

/-- The total number of balls in the bag -/
def total_balls : Nat := white_balls + red_balls

/-- The probability of drawing 2 or fewer white balls before a red ball -/
def prob_two_or_fewer_white : Rat :=
  (red_balls : Rat) / total_balls +
  (white_balls * red_balls : Rat) / (total_balls * (total_balls - 1)) +
  (white_balls * (white_balls - 1) * red_balls : Rat) / (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_two_or_fewer_white_eq_23_28 : prob_two_or_fewer_white = 23 / 28 := by
  sorry

end prob_two_or_fewer_white_eq_23_28_l1160_116076


namespace smallest_consecutive_triangle_perimeter_l1160_116031

/-- A triangle with consecutive integer side lengths. -/
structure ConsecutiveTriangle where
  a : ℕ
  valid : a > 0

/-- The three side lengths of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.sides (t : ConsecutiveTriangle) : Fin 3 → ℕ
  | 0 => t.a
  | 1 => t.a + 1
  | 2 => t.a + 2

/-- The perimeter of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.perimeter (t : ConsecutiveTriangle) : ℕ :=
  3 * t.a + 3

/-- Predicate for whether a ConsecutiveTriangle satisfies the Triangle Inequality. -/
def ConsecutiveTriangle.satisfiesTriangleInequality (t : ConsecutiveTriangle) : Prop :=
  t.sides 0 + t.sides 1 > t.sides 2 ∧
  t.sides 0 + t.sides 2 > t.sides 1 ∧
  t.sides 1 + t.sides 2 > t.sides 0

/-- The smallest ConsecutiveTriangle that satisfies the Triangle Inequality. -/
def smallestValidConsecutiveTriangle : ConsecutiveTriangle :=
  { a := 2
    valid := by simp }

/-- Theorem: The smallest possible perimeter of a triangle with consecutive integer side lengths is 9. -/
theorem smallest_consecutive_triangle_perimeter :
  (∀ t : ConsecutiveTriangle, t.satisfiesTriangleInequality → t.perimeter ≥ 9) ∧
  smallestValidConsecutiveTriangle.satisfiesTriangleInequality ∧
  smallestValidConsecutiveTriangle.perimeter = 9 :=
sorry

end smallest_consecutive_triangle_perimeter_l1160_116031


namespace planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l1160_116080

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two different planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (l : Line) (p1 p2 : Plane) (h1 : p1 ≠ p2)
  (h2 : perpendicular_plane_line p1 l) (h3 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two different lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line) (h1 : l1 ≠ l2)
  (h2 : perpendicular_line_plane l1 p) (h3 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

end planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l1160_116080


namespace power_product_equality_l1160_116070

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end power_product_equality_l1160_116070


namespace parabola_intersection_l1160_116034

theorem parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 4 * y + 7) → k = 25/3 := by
  sorry

end parabola_intersection_l1160_116034


namespace potato_bag_weight_l1160_116083

/-- The weight of each bag of potatoes -/
def bag_weight (total_potatoes damaged_potatoes : ℕ) (price_per_bag total_revenue : ℚ) : ℚ :=
  (total_potatoes - damaged_potatoes) * price_per_bag / total_revenue

/-- Theorem stating the weight of each bag of potatoes -/
theorem potato_bag_weight :
  bag_weight 6500 150 72 9144 = 50 := by
  sorry

end potato_bag_weight_l1160_116083


namespace lowest_price_breaks_even_l1160_116085

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_breaks_even 
  (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_components : ℕ) :
  let price := lowest_price_per_component production_cost shipping_cost fixed_costs num_components
  (price * num_components : ℚ) = (production_cost + shipping_cost) * num_components + fixed_costs :=
by sorry

#eval lowest_price_per_component 80 5 16500 150

end lowest_price_breaks_even_l1160_116085


namespace stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l1160_116011

/- Define the associations and their sizes -/
def associations : Fin 3 → ℕ
| 0 => 27  -- Association A
| 1 => 9   -- Association B
| 2 => 18  -- Association C

/- Total number of athletes -/
def total_athletes : ℕ := (associations 0) + (associations 1) + (associations 2)

/- Number of athletes to be selected -/
def selected_athletes : ℕ := 6

/- Theorem for stratified sampling -/
theorem stratified_sampling_proportion (i : Fin 3) :
  (associations i) * selected_athletes = (associations i) * total_athletes / total_athletes :=
sorry

/- Theorem for number of ways to choose 2 from 6 -/
theorem choose_two_from_six :
  Nat.choose selected_athletes 2 = 15 :=
sorry

/- Theorem for probability of selecting at least one from last two -/
theorem prob_at_least_one_from_last_two :
  (Nat.choose 4 1 * Nat.choose 2 1 + Nat.choose 2 2) / Nat.choose 6 2 = 3 / 5 :=
sorry

end stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l1160_116011


namespace train_bridge_crossing_time_l1160_116073

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (bridge_length : ℝ)
  (train_length : ℝ)
  (lamp_post_time : ℝ)
  (h1 : bridge_length = 200)
  (h2 : train_length = 200)
  (h3 : lamp_post_time = 5)
  : ℝ :=
by
  -- The time taken for the train to cross the bridge is 10 seconds
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l1160_116073


namespace disk_space_remaining_l1160_116004

/-- Calculates the remaining disk space given total space and used space -/
def remaining_space (total : ℕ) (used : ℕ) : ℕ :=
  total - used

/-- Theorem: Given 28 GB total space and 26 GB used space, the remaining space is 2 GB -/
theorem disk_space_remaining :
  remaining_space 28 26 = 2 := by
  sorry

end disk_space_remaining_l1160_116004


namespace apple_pear_cost_l1160_116095

theorem apple_pear_cost (x y : ℝ) 
  (eq1 : x + 2*y = 194) 
  (eq2 : 2*x + 5*y = 458) : 
  x = 54 ∧ y = 70 := by
  sorry

end apple_pear_cost_l1160_116095


namespace max_area_triangle_l1160_116002

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x + Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x - Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem max_area_triangle (C : ℝ) (a b c : ℝ) (hf : f C = 2) (hc : c = Real.sqrt 3) :
  ∃ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∧ 
  (∀ S' : ℝ, S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
sorry

end max_area_triangle_l1160_116002


namespace constant_term_expansion_l1160_116050

theorem constant_term_expansion :
  let f := fun (x : ℝ) => (x - 1/x)^6
  ∃ (c : ℝ), c = -20 ∧ 
    ∀ (x : ℝ), x ≠ 0 → (∃ (g : ℝ → ℝ), f x = c + x * g x + (1/x) * g (1/x)) :=
by sorry

end constant_term_expansion_l1160_116050


namespace parabola_directrix_l1160_116089

/-- The directrix of a parabola y = -x^2 --/
theorem parabola_directrix : ∃ (d : ℝ), ∀ (x y : ℝ),
  y = -x^2 → (∃ (p : ℝ × ℝ), (x - p.1)^2 + (y - p.2)^2 = (y - d)^2 ∧ y ≤ d) → d = 1/4 := by
  sorry

end parabola_directrix_l1160_116089


namespace quadratic_roots_transformation_l1160_116010

theorem quadratic_roots_transformation (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 ↔ x = x₁^2 + x₂^2 ∨ x = 2 * x₁ * x₂) :=
by sorry

end quadratic_roots_transformation_l1160_116010


namespace equation_one_solution_equation_two_no_solution_l1160_116059

-- Equation 1
theorem equation_one_solution :
  ∃! x : ℚ, (x / (2 * x - 1)) + (2 / (1 - 2 * x)) = 3 :=
by sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ x : ℚ, (4 / (x^2 - 4)) - (1 / (x - 2)) = 0 :=
by sorry

end equation_one_solution_equation_two_no_solution_l1160_116059


namespace subtract_negative_l1160_116052

theorem subtract_negative (a b : ℝ) : a - (-b) = a + b := by
  sorry

end subtract_negative_l1160_116052


namespace tangent_line_inclination_angle_l1160_116037

/-- The curve y = x³ - 2x + 4 has a tangent line at (1, 3) with an inclination angle of 45° -/
theorem tangent_line_inclination_angle :
  let f (x : ℝ) := x^3 - 2*x + 4
  let f' (x : ℝ) := 3*x^2 - 2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  (f x₀ = y₀) ∧ 
  (Real.tan θ = f' x₀) →
  θ = Real.pi / 4 := by
sorry

end tangent_line_inclination_angle_l1160_116037


namespace probability_failed_math_given_failed_chinese_l1160_116079

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16)
  (h2 : failed_chinese = 0.07)
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 := by sorry

end probability_failed_math_given_failed_chinese_l1160_116079


namespace olivias_bags_l1160_116009

/-- The number of cans Olivia had in total -/
def total_cans : ℕ := 20

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 5

/-- The number of bags Olivia had -/
def number_of_bags : ℕ := total_cans / cans_per_bag

theorem olivias_bags : number_of_bags = 4 := by
  sorry

end olivias_bags_l1160_116009


namespace intersection_of_A_and_B_l1160_116091

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {y | ∃ x, y = Real.exp x + 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l1160_116091


namespace expression_evaluation_l1160_116027

theorem expression_evaluation (x : ℝ) (h : x = -2) : 
  (3 * x / (x - 1) - x / (x + 1)) * (x^2 - 1) / x = 0 := by
  sorry

end expression_evaluation_l1160_116027


namespace city_population_ratio_l1160_116043

theorem city_population_ratio (x y z : ℕ) 
  (h1 : y = 2 * z)
  (h2 : x = 12 * z) :
  x / y = 6 := by
  sorry

end city_population_ratio_l1160_116043


namespace incorrect_statement_l1160_116081

def U : Finset Nat := {1, 2, 3, 4}
def M : Finset Nat := {1, 2}
def N : Finset Nat := {2, 4}

theorem incorrect_statement : M ∩ (U \ N) ≠ {1, 2, 3} := by sorry

end incorrect_statement_l1160_116081


namespace inverse_function_range_l1160_116088

/-- Given a function f and its inverse, prove the range of a -/
theorem inverse_function_range (a : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : 
  (∀ x, f x = a^(x+1) - 2) →
  (a > 1) →
  (∀ x, f_inv (f x) = x) →
  (∀ x, x ≤ 0 → f_inv x ≤ 0) →
  a ≥ 2 :=
by sorry

end inverse_function_range_l1160_116088


namespace travelers_speed_l1160_116024

/-- Given two travelers A and B, where B travels 2 km/h faster than A,
    and they meet after 3 hours having traveled a total of 24 km,
    prove that A's speed is 3 km/h. -/
theorem travelers_speed (x : ℝ) : 3*x + 3*(x + 2) = 24 → x = 3 := by
  sorry

end travelers_speed_l1160_116024


namespace monotone_quadratic_function_m_range_l1160_116064

/-- A function f is monotonically increasing on an interval (a, b) if for any x, y in (a, b) with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- The function f(x) = mx^2 + x - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x - 1

theorem monotone_quadratic_function_m_range :
  (∀ m : ℝ, MonotonicallyIncreasing (f m) (-1) Real.pi) ↔ 
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1/2) :=
sorry

end monotone_quadratic_function_m_range_l1160_116064


namespace gcd_and_binary_conversion_l1160_116084

theorem gcd_and_binary_conversion :
  (Nat.gcd 153 119 = 17) ∧
  (ToString.toString (Nat.toDigits 2 89) = "1011001") := by
  sorry

end gcd_and_binary_conversion_l1160_116084


namespace sin_2005_equals_neg_sin_25_l1160_116056

theorem sin_2005_equals_neg_sin_25 :
  Real.sin (2005 * π / 180) = -Real.sin (25 * π / 180) := by
  sorry

end sin_2005_equals_neg_sin_25_l1160_116056


namespace b_fourth_zero_implies_b_squared_zero_l1160_116025

theorem b_fourth_zero_implies_b_squared_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end b_fourth_zero_implies_b_squared_zero_l1160_116025


namespace exists_unique_subset_l1160_116029

theorem exists_unique_subset : ∃ (S : Set ℤ), 
  ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ S ∧ pair.2 ∈ S ∧ n = 2 * pair.1 + pair.2 := by
  sorry

end exists_unique_subset_l1160_116029


namespace parallel_vectors_m_value_l1160_116060

/-- 
Given two vectors a and b in R^2, where a = (1, m) and b = (-1, 2m+1),
prove that if a and b are parallel, then m = -1/3.
-/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 2*m+1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = -1/3 := by
sorry

end parallel_vectors_m_value_l1160_116060


namespace geometric_sequence_sum_l1160_116006

theorem geometric_sequence_sum (a b c q : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) = (a + b + c) * q ∧
  (c + a - b) = (a + b + c) * q^2 ∧
  (a + b - c) = (a + b + c) * q^3 →
  q^3 + q^2 + q = 1 := by
  sorry

end geometric_sequence_sum_l1160_116006


namespace min_value_of_f_l1160_116094

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m ∧ f x m = -6 :=
by sorry

end min_value_of_f_l1160_116094


namespace triangle_perimeter_is_twelve_l1160_116048

/-- The line equation in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The triangle formed by a line and the coordinate axes -/
structure Triangle where
  line : Line

def Triangle.perimeter (t : Triangle) : ℝ :=
  sorry

theorem triangle_perimeter_is_twelve (t : Triangle) :
  t.line = { a := 1/3, b := 1/4, c := 1 } →
  t.perimeter = 12 :=
sorry

end triangle_perimeter_is_twelve_l1160_116048


namespace total_toys_l1160_116066

/-- Given that Annie has three times more toys than Mike, Annie has two less toys than Tom,
    and Mike has 6 toys, prove that the total number of toys Annie, Mike, and Tom have is 56. -/
theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ)
  (h1 : annie_toys = 3 * mike_toys)
  (h2 : tom_toys = annie_toys + 2)
  (h3 : mike_toys = 6) :
  annie_toys + mike_toys + tom_toys = 56 :=
by sorry

end total_toys_l1160_116066


namespace corrected_mean_specific_corrected_mean_l1160_116082

/-- Given a set of observations with an incorrect entry, calculate the corrected mean -/
theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n > 0 →
  let total_sum := n * original_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = (n * original_mean - incorrect_value + correct_value) / n :=
by sorry

/-- The specific problem instance -/
theorem specific_corrected_mean :
  let n : ℕ := 40
  let original_mean : ℝ := 100
  let incorrect_value : ℝ := 75
  let correct_value : ℝ := 50
  (n * original_mean - incorrect_value + correct_value) / n = 99.375 :=
by sorry

end corrected_mean_specific_corrected_mean_l1160_116082


namespace factories_unchecked_l1160_116092

theorem factories_unchecked (total : ℕ) (group1 : ℕ) (group2 : ℕ) 
  (h1 : total = 169) 
  (h2 : group1 = 69) 
  (h3 : group2 = 52) : 
  total - (group1 + group2) = 48 := by
  sorry

end factories_unchecked_l1160_116092


namespace no_xy_term_when_k_is_3_l1160_116017

/-- The polynomial that we're analyzing -/
def polynomial (x y k : ℝ) : ℝ := -x^2 - 3*k*x*y - 3*y^2 + 9*x*y - 8

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := -3*k + 9

theorem no_xy_term_when_k_is_3 :
  ∃ (k : ℝ), xy_coefficient k = 0 ∧ k = 3 :=
sorry

end no_xy_term_when_k_is_3_l1160_116017


namespace existence_of_special_numbers_l1160_116013

theorem existence_of_special_numbers : ∃ (a b c : ℕ), 
  (a > 10^10 ∧ b > 10^10 ∧ c > 10^10) ∧
  (a * b * c) % (a + 2012) = 0 ∧
  (a * b * c) % (b + 2012) = 0 ∧
  (a * b * c) % (c + 2012) = 0 :=
by sorry

end existence_of_special_numbers_l1160_116013


namespace smallest_n_for_sqrt_inequality_l1160_116003

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.01 ↔ n ≥ 2501) :=
by sorry

end smallest_n_for_sqrt_inequality_l1160_116003


namespace lcm_6_15_l1160_116069

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end lcm_6_15_l1160_116069


namespace root_implies_p_minus_q_l1160_116008

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (p q : ℝ) (x : ℂ) : Prop :=
  2 * x^2 + p * x + q = 0

-- State the theorem
theorem root_implies_p_minus_q (p q : ℝ) :
  equation p q (-2 * i - 3) → p - q = -14 := by
  sorry

end root_implies_p_minus_q_l1160_116008


namespace intersection_when_a_2_B_subset_A_condition_l1160_116045

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1: Intersection when a = 2
theorem intersection_when_a_2 : 
  A 2 ∩ B 2 = {x : ℝ | 4 < x ∧ x < 5} := by sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_condition (a : ℝ) :
  a ≠ 1 →
  (B a ⊆ A a ↔ (1 < a ∧ a ≤ 3) ∨ a = -1) := by sorry

end intersection_when_a_2_B_subset_A_condition_l1160_116045


namespace parabola_shift_down_2_l1160_116042

/-- The equation of a parabola after vertical shift -/
def shifted_parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b

/-- Theorem: Shifting y = x^2 down by 2 units results in y = x^2 - 2 -/
theorem parabola_shift_down_2 :
  shifted_parabola 1 (-2) = λ x => x^2 - 2 := by
  sorry

end parabola_shift_down_2_l1160_116042


namespace nested_fraction_equality_l1160_116068

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end nested_fraction_equality_l1160_116068


namespace hcf_of_210_and_605_l1160_116054

theorem hcf_of_210_and_605 :
  let a := 210
  let b := 605
  let lcm_ab := 2310
  lcm a b = lcm_ab →
  Nat.gcd a b = 55 := by
sorry

end hcf_of_210_and_605_l1160_116054


namespace printer_ratio_l1160_116036

theorem printer_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx_time : x = 16) (hy_time : y = 12) (hz_time : z = 8) :
  x / ((1 / y + 1 / z)⁻¹) = 10 / 3 := by
  sorry

end printer_ratio_l1160_116036


namespace floor_ceiling_identity_l1160_116035

theorem floor_ceiling_identity (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ⌊x⌋ + x - ⌈x⌉ = x - 1 := by
  sorry

end floor_ceiling_identity_l1160_116035


namespace jose_maria_age_difference_jose_maria_age_difference_proof_l1160_116007

theorem jose_maria_age_difference : ℕ → ℕ → Prop :=
  fun jose_age maria_age =>
    (jose_age > maria_age) →
    (jose_age + maria_age = 40) →
    (maria_age = 14) →
    (jose_age - maria_age = 12)

-- The proof would go here, but we'll skip it as requested
theorem jose_maria_age_difference_proof : ∃ (j m : ℕ), jose_maria_age_difference j m :=
  sorry

end jose_maria_age_difference_jose_maria_age_difference_proof_l1160_116007


namespace square_of_negative_square_l1160_116061

theorem square_of_negative_square (a : ℝ) : (-a^2)^2 = a^4 := by
  sorry

end square_of_negative_square_l1160_116061


namespace work_completion_time_l1160_116055

/-- Proves that A can complete the work in 15 days given the conditions -/
theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 20) = 1 - 0.5333333333333333) →  -- Condition after 4 days of joint work
  x = 15 := by
  sorry

end work_completion_time_l1160_116055


namespace european_stamps_count_l1160_116063

/-- Represents the number of stamps from Asian countries -/
def asian_stamps : ℕ := sorry

/-- Represents the number of stamps from European countries -/
def european_stamps : ℕ := sorry

/-- The total number of stamps Jesse has -/
def total_stamps : ℕ := 444

/-- European stamps are three times the number of Asian stamps -/
axiom european_triple_asian : european_stamps = 3 * asian_stamps

/-- The sum of Asian and European stamps equals the total stamps -/
axiom sum_equals_total : asian_stamps + european_stamps = total_stamps

/-- Theorem stating that the number of European stamps is 333 -/
theorem european_stamps_count : european_stamps = 333 := by sorry

end european_stamps_count_l1160_116063


namespace polygon_perimeter_bounds_l1160_116041

theorem polygon_perimeter_bounds :
  ∃ (m₃ m₄ m₅ m₆ m₇ m₈ m₉ m₁₀ : ℝ),
    (abs m₃ ≤ 3) ∧
    (abs m₄ ≤ 5) ∧
    (abs m₅ ≤ 7) ∧
    (abs m₆ ≤ 9) ∧
    (abs m₇ ≤ 12) ∧
    (abs m₈ ≤ 14) ∧
    (abs m₉ ≤ 16) ∧
    (abs m₁₀ ≤ 19) ∧
    (m₃ ≤ m₄) ∧ (m₄ ≤ m₅) ∧ (m₅ ≤ m₆) ∧ (m₆ ≤ m₇) ∧
    (m₇ ≤ m₈) ∧ (m₈ ≤ m₉) ∧ (m₉ ≤ m₁₀) := by
  sorry


end polygon_perimeter_bounds_l1160_116041


namespace tangent_circle_existence_l1160_116018

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line as a pair of points
structure Line where
  point1 : Point
  point2 : Point

-- Define tangency between two circles at a point
def CircleTangentToCircle (S S' : Circle) (A : Point) : Prop :=
  -- The centers of S and S' and point A are collinear
  sorry

-- Define tangency between a circle and a line at a point
def CircleTangentToLine (S : Circle) (l : Line) (B : Point) : Prop :=
  -- The radius of S at B is perpendicular to l
  sorry

-- The main theorem
theorem tangent_circle_existence 
  (S : Circle) (A : Point) (l : Line) : 
  ∃ (S' : Circle) (B : Point), 
    CircleTangentToCircle S S' A ∧ 
    CircleTangentToLine S' l B :=
  sorry

end tangent_circle_existence_l1160_116018


namespace complement_M_intersect_N_l1160_116062

def U : Set ℕ := {x | x > 0 ∧ x < 9}

def M : Set ℕ := {1, 2, 3}

def N : Set ℕ := {3, 4, 5, 6}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end complement_M_intersect_N_l1160_116062


namespace jason_music_store_expenditure_l1160_116030

/-- The total cost of Jason's music store purchases --/
def total_cost : ℚ :=
  142.46 + 8.89 + 7.00 + 15.75 + 12.95 + 36.50 + 5.25

/-- Theorem stating that Jason's total music store expenditure is $229.80 --/
theorem jason_music_store_expenditure :
  total_cost = 229.80 := by sorry

end jason_music_store_expenditure_l1160_116030


namespace pedestrian_speeds_l1160_116033

theorem pedestrian_speeds (x y : ℝ) 
  (h1 : x + y = 14)
  (h2 : (3/2) * x + (1/2) * y = 13) :
  (x = 6 ∧ y = 8) ∨ (x = 8 ∧ y = 6) := by
  sorry

end pedestrian_speeds_l1160_116033


namespace max_integer_value_of_fraction_l1160_116093

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4*x^2 + 12*x + 23) / (4*x^2 + 12*x + 9) ≤ 8 ∧ 
  ∃ y : ℝ, (4*y^2 + 12*y + 23) / (4*y^2 + 12*y + 9) > 7 := by
  sorry

end max_integer_value_of_fraction_l1160_116093


namespace unique_stamp_solution_l1160_116058

/-- Given a positive integer n, returns true if 120 cents is the greatest
    postage that cannot be formed using stamps of 9, n, and n+2 cents -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  (∀ k : ℕ, k ≤ 120 → ¬∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k) ∧
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k)

/-- The only positive integer n that satisfies the stamp condition is 17 -/
theorem unique_stamp_solution :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 17 :=
sorry

end unique_stamp_solution_l1160_116058


namespace sheep_with_only_fleas_l1160_116001

theorem sheep_with_only_fleas (total : ℕ) (lice : ℕ) (both : ℕ) (only_fleas : ℕ) : 
  total = 2 * lice →
  both = 84 →
  lice = 94 →
  total = only_fleas + (lice - both) + both →
  only_fleas = 94 := by
sorry

end sheep_with_only_fleas_l1160_116001


namespace root_sum_squares_l1160_116019

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 20*a^2 + 18*a - 7 = 0) →
  (b^3 - 20*b^2 + 18*b - 7 = 0) →
  (c^3 - 20*c^2 + 18*c - 7 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 764 := by
sorry

end root_sum_squares_l1160_116019


namespace terminal_side_in_second_quadrant_l1160_116078

/-- Given that α = 3, prove that the terminal side of α lies in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : ℝ) (h : α = 3) :
  (π / 2 : ℝ) < α ∧ α < π :=
sorry

end terminal_side_in_second_quadrant_l1160_116078


namespace box_removal_proof_l1160_116074

theorem box_removal_proof (total_boxes : Nat) (boxes_10lb boxes_20lb boxes_30lb boxes_40lb : Nat)
  (initial_avg_weight : Nat) (target_avg_weight : Nat) 
  (h1 : total_boxes = 30)
  (h2 : boxes_10lb = 10)
  (h3 : boxes_20lb = 10)
  (h4 : boxes_30lb = 5)
  (h5 : boxes_40lb = 5)
  (h6 : initial_avg_weight = 20)
  (h7 : target_avg_weight = 17) :
  let total_weight := boxes_10lb * 10 + boxes_20lb * 20 + boxes_30lb * 30 + boxes_40lb * 40
  let remaining_boxes := total_boxes - 6
  let remaining_weight := total_weight - (5 * 20 + 1 * 40)
  remaining_weight / remaining_boxes = target_avg_weight :=
by sorry

end box_removal_proof_l1160_116074


namespace marie_task_completion_time_l1160_116096

-- Define the start time of the first task
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 9 * 60 + 20  -- 9:20 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem marie_task_completion_time :
  let total_time_two_tasks := end_second_task - start_time
  let task_duration := total_time_two_tasks / 2
  let completion_time := end_second_task + 2 * task_duration
  completion_time = 11 * 60 + 40  -- 11:40 AM in minutes since midnight
:= by sorry

end marie_task_completion_time_l1160_116096


namespace only_one_correct_statement_l1160_116032

/-- Represents the confidence level in the study conclusion -/
def confidence_level : ℝ := 0.99

/-- Represents the four statements about smoking and lung cancer -/
inductive Statement
  | all_smokers_have_cancer
  | high_probability_of_cancer
  | some_smokers_have_cancer
  | possibly_no_smokers_have_cancer

/-- Determines if a statement is correct given the confidence level -/
def is_correct (s : Statement) (conf : ℝ) : Prop :=
  match s with
  | Statement.possibly_no_smokers_have_cancer => conf < 1
  | _ => False

/-- The main theorem stating that only one statement is correct -/
theorem only_one_correct_statement : 
  (∃! s : Statement, is_correct s confidence_level) ∧ 
  (is_correct Statement.possibly_no_smokers_have_cancer confidence_level) :=
sorry

end only_one_correct_statement_l1160_116032


namespace sum_of_solutions_l1160_116071

theorem sum_of_solutions (x : ℝ) : 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b) → 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b ∧ a + b = 7/6) :=
by sorry

end sum_of_solutions_l1160_116071


namespace fractional_equation_root_l1160_116099

/-- If the equation (3 / (x - 4)) + ((x + m) / (4 - x)) = 1 has a root, then m = -1 -/
theorem fractional_equation_root (x m : ℚ) : 
  (∃ x, (3 / (x - 4)) + ((x + m) / (4 - x)) = 1) → m = -1 := by
  sorry

end fractional_equation_root_l1160_116099


namespace relationship_abc_l1160_116053

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end relationship_abc_l1160_116053


namespace wrong_mark_calculation_l1160_116026

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg correct_mark : ℝ) : 
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 96 ∧ 
  correct_mark = 10 → 
  ∃ wrong_mark : ℝ, 
    wrong_mark = 50 ∧ 
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    n * correct_avg = (n - 1) * correct_avg + correct_mark :=
by sorry

end wrong_mark_calculation_l1160_116026


namespace sea_horse_count_l1160_116021

theorem sea_horse_count : 
  ∀ (s p : ℕ), 
  (s : ℚ) / p = 5 / 11 → 
  p = s + 85 → 
  s = 70 := by
sorry

end sea_horse_count_l1160_116021


namespace inheritance_calculation_inheritance_value_l1160_116086

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 49655

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The total tax paid in dollars -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax_paid := by sorry

theorem inheritance_value :
  inheritance = 49655 := by sorry

end inheritance_calculation_inheritance_value_l1160_116086


namespace bus_speed_problem_l1160_116065

theorem bus_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 72 →
  speed_ratio = 1.2 →
  time_difference = 1/5 →
  ∀ (speed_large : ℝ),
    (distance / speed_large - distance / (speed_ratio * speed_large) = time_difference) →
    speed_large = 60 := by
  sorry

end bus_speed_problem_l1160_116065


namespace intersection_of_A_and_B_l1160_116023

def set_A : Set ℝ := {x | 2 * x + 1 > 0}
def set_B : Set ℝ := {x | |x - 1| < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | -1/2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1160_116023


namespace speed_increase_percentage_l1160_116098

theorem speed_increase_percentage (distance : ℝ) (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  distance = 96 →
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 50 ∧
    distance / (current_speed * (1 + increase_percentage / 100)) = 
    distance / (current_speed - speed_reduction) - time_difference :=
by sorry

end speed_increase_percentage_l1160_116098


namespace projection_vector_l1160_116044

/-- Two parallel lines r and s in 2D space -/
structure ParallelLines where
  r : ℝ → ℝ × ℝ
  s : ℝ → ℝ × ℝ
  hr : ∀ t, r t = (2 + 5*t, 3 - 2*t)
  hs : ∀ u, s u = (1 + 5*u, -2 - 2*u)

/-- Points C, D, and Q in 2D space -/
structure Points (l : ParallelLines) where
  C : ℝ × ℝ
  D : ℝ × ℝ
  Q : ℝ × ℝ
  hC : ∃ t, l.r t = C
  hD : ∃ u, l.s u = D
  hQ : (Q.1 - C.1) * 5 + (Q.2 - C.2) * (-2) = 0 -- Q is on the perpendicular to s passing through C

/-- The theorem to be proved -/
theorem projection_vector (l : ParallelLines) (p : Points l) :
  ∃ k : ℝ, 
    (p.Q.1 - p.C.1, p.Q.2 - p.C.2) = k • (-2, -5) ∧
    (p.D.1 - p.C.1) * (-2) + (p.D.2 - p.C.2) * (-5) = 
      (p.Q.1 - p.C.1) * (-2) + (p.Q.2 - p.C.2) * (-5) ∧
    -2 - (-5) = 3 :=
  sorry

end projection_vector_l1160_116044


namespace bulb_arrangement_count_l1160_116087

/-- The number of ways to arrange bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem stating the number of arrangements for the given bulb counts -/
theorem bulb_arrangement_count :
  bulb_arrangements 7 6 10 = 1717716 := by
  sorry

end bulb_arrangement_count_l1160_116087
