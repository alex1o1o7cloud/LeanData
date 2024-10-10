import Mathlib

namespace inequality_solution_existence_l1970_197031

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x * Real.log x - a < 0) ↔ a > -1 / Real.exp 1 := by
  sorry

end inequality_solution_existence_l1970_197031


namespace fourth_person_height_l1970_197007

/-- Represents the heights of four people standing in order of increasing height. -/
structure HeightGroup where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the height problem. -/
def height_conditions (h : HeightGroup) : Prop :=
  h.second = h.first + 2 ∧
  h.third = h.second + 2 ∧
  h.fourth = h.third + 6 ∧
  (h.first + h.second + h.third + h.fourth) / 4 = 79

/-- The theorem stating that under the given conditions, the fourth person is 85 inches tall. -/
theorem fourth_person_height (h : HeightGroup) :
  height_conditions h → h.fourth = 85 := by
  sorry

end fourth_person_height_l1970_197007


namespace arithmetic_calculation_l1970_197041

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end arithmetic_calculation_l1970_197041


namespace cookies_eaten_difference_l1970_197073

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) 
  (h1 : initial_sweet = 37)
  (h2 : initial_salty = 11)
  (h3 : eaten_sweet = 5)
  (h4 : eaten_salty = 2) :
  eaten_sweet - eaten_salty = 3 :=
by
  sorry

end cookies_eaten_difference_l1970_197073


namespace dan_remaining_limes_l1970_197013

/-- The number of limes Dan has after giving some to Sara -/
def limes_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 5 limes remaining -/
theorem dan_remaining_limes :
  limes_remaining 9 4 = 5 := by
  sorry

end dan_remaining_limes_l1970_197013


namespace mistake_correction_l1970_197074

theorem mistake_correction (x : ℤ) : x - 23 = 4 → x * 23 = 621 := by
  sorry

end mistake_correction_l1970_197074


namespace sum_of_coefficients_3x_minus_4y_power_20_l1970_197000

theorem sum_of_coefficients_3x_minus_4y_power_20 :
  let f : ℝ → ℝ → ℝ := λ x y => (3*x - 4*y)^20
  (f 1 1) = 1 :=
by sorry

end sum_of_coefficients_3x_minus_4y_power_20_l1970_197000


namespace product_of_roots_l1970_197089

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end product_of_roots_l1970_197089


namespace equation_solution_l1970_197068

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (2 * x + 4) / (x^2 + 4 * x - 5) = (2 - x) / (x - 1) ∧ x = -6 := by
  sorry

end equation_solution_l1970_197068


namespace z_modulus_l1970_197090

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition z(i+1) = i
def condition (z : ℂ) : Prop := z * (i + 1) = i

-- State the theorem
theorem z_modulus (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end z_modulus_l1970_197090


namespace diagonal_cubes_140_320_360_l1970_197022

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The internal diagonal of a 140 × 320 × 360 rectangular solid passes through 760 unit cubes -/
theorem diagonal_cubes_140_320_360 :
  diagonal_cubes 140 320 360 = 760 := by
  sorry

end diagonal_cubes_140_320_360_l1970_197022


namespace unique_solution_l1970_197058

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x ^ (floor x) = 9 / 2

-- State the theorem
theorem unique_solution : 
  ∃! x : ℝ, equation x ∧ x = (3 * Real.sqrt 2) / 2 :=
sorry

end unique_solution_l1970_197058


namespace base_six_units_digit_l1970_197036

theorem base_six_units_digit : 
  (123 * 78 - 156) % 6 = 0 := by sorry

end base_six_units_digit_l1970_197036


namespace equal_diff_squares_properties_l1970_197047

-- Definition of "equal difference of squares sequence"
def is_equal_diff_squares (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = p

theorem equal_diff_squares_properties :
  -- Statement 1
  is_equal_diff_squares (fun n => (-1) ^ n) ∧
  -- Statement 2
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    ∃ d : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = d) ∧
  -- Statement 3
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    (∃ d : ℝ, ∀ n ≥ 2, a n - a (n - 1) = d) →
    ∃ c : ℝ, ∀ n, a n = c) ∧
  -- Statement 4
  ∃ a : ℕ → ℝ, is_equal_diff_squares a ∧
    ∀ k : ℕ+, is_equal_diff_squares (fun n => a (k * n)) :=
by sorry

end equal_diff_squares_properties_l1970_197047


namespace cistern_water_breadth_l1970_197051

/-- Proves that for a cistern with given dimensions and wet surface area, 
    the breadth of water is 1.25 meters. -/
theorem cistern_water_breadth 
  (length : ℝ) 
  (width : ℝ) 
  (wet_surface_area : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 4) 
  (h_wet_area : wet_surface_area = 49) : 
  ∃ (breadth : ℝ), 
    breadth = 1.25 ∧ 
    wet_surface_area = length * width + 2 * (length + width) * breadth :=
by sorry

end cistern_water_breadth_l1970_197051


namespace system_solution_l1970_197052

theorem system_solution (x y z : ℝ) : 
  x * y = 15 - 3 * x - 2 * y →
  y * z = 8 - 2 * y - 4 * z →
  x * z = 56 - 5 * x - 6 * z →
  x > 0 →
  x = 8 := by
sorry

end system_solution_l1970_197052


namespace largest_multiple_in_sequence_l1970_197098

theorem largest_multiple_in_sequence : 
  ∀ (n : ℕ), 
  (3*n + 3*(n+1) + 3*(n+2) = 117) → 
  (max (3*n) (max (3*(n+1)) (3*(n+2))) = 42) := by
sorry

end largest_multiple_in_sequence_l1970_197098


namespace abc_fraction_value_l1970_197076

theorem abc_fraction_value (a b c : ℕ+) 
  (h : a^2*b + b^2*c + a*c^2 + a + b + c = 2*(a*b + b*c + a*c)) :
  (c : ℚ)^2017 / ((a : ℚ)^2016 + (b : ℚ)^2018) = 1/2 := by
sorry

end abc_fraction_value_l1970_197076


namespace sum_of_squares_residuals_l1970_197046

/-- Linear Regression Sum of Squares -/
structure LinearRegressionSS where
  SST : ℝ  -- Total sum of squares
  SSR : ℝ  -- Sum of squares due to regression
  SSE : ℝ  -- Sum of squares for residuals

/-- Theorem: Sum of Squares for Residuals in Linear Regression -/
theorem sum_of_squares_residuals 
  (lr : LinearRegressionSS) 
  (h1 : lr.SST = 13) 
  (h2 : lr.SSR = 10) 
  (h3 : lr.SST = lr.SSR + lr.SSE) : 
  lr.SSE = 3 := by
sorry

end sum_of_squares_residuals_l1970_197046


namespace symmetry_correctness_l1970_197048

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry operations in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetry_yoz_plane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetry_y_axis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetry_origin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

/-- The theorem to be proved -/
theorem symmetry_correctness (a b c : ℝ) : 
  let M : Point3D := ⟨a, b, c⟩
  (symmetry_x_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_yoz_plane M ≠ ⟨a, -b, -c⟩) ∧ 
  (symmetry_y_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_origin M = ⟨-a, -b, -c⟩) := by
  sorry

end symmetry_correctness_l1970_197048


namespace yummy_kibble_percentage_proof_l1970_197071

/-- The number of vets in the state -/
def total_vets : ℕ := 1000

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℚ := 20 / 100

/-- The number of additional vets recommending Yummy Dog Kibble compared to Puppy Kibble -/
def additional_yummy_kibble_vets : ℕ := 100

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℚ := 30 / 100

theorem yummy_kibble_percentage_proof :
  (puppy_kibble_percentage * total_vets + additional_yummy_kibble_vets : ℚ) / total_vets = yummy_kibble_percentage := by
  sorry

end yummy_kibble_percentage_proof_l1970_197071


namespace probability_one_of_each_color_l1970_197067

def total_marbles : ℕ := 12
def marbles_per_color : ℕ := 3
def colors : ℕ := 4
def selected_marbles : ℕ := 4

/-- The probability of selecting one marble of each color when randomly selecting 4 marbles
    without replacement from a bag containing 3 red, 3 blue, 3 green, and 3 yellow marbles. -/
theorem probability_one_of_each_color : 
  (marbles_per_color ^ colors : ℚ) / (total_marbles.choose selected_marbles) = 9 / 55 := by
  sorry

end probability_one_of_each_color_l1970_197067


namespace parabola_equation_l1970_197020

/-- Given a point A(1,1) and a parabola C: y^2 = 2px (p > 0) whose focus lies on the perpendicular
    bisector of OA, prove that the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (1, 1)
  let O : ℝ × ℝ := (0, 0)
  let perpendicular_bisector := {(x, y) : ℝ × ℝ | x + y = 1}
  let focus : ℝ × ℝ := (p / 2, 0)
  focus ∈ perpendicular_bisector →
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x :=
by sorry

end parabola_equation_l1970_197020


namespace benjamin_weekly_miles_l1970_197049

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked (work_distance : ℕ) (dog_walk_distance : ℕ) (friend_distance : ℕ) (store_distance : ℕ) : ℕ :=
  let work_trips := 2 * work_distance * 5
  let dog_walks := 2 * dog_walk_distance * 7
  let friend_visit := 2 * friend_distance
  let store_trips := 2 * store_distance * 2
  work_trips + dog_walks + friend_visit + store_trips

/-- Theorem stating that Benjamin walks 102 miles in a week --/
theorem benjamin_weekly_miles :
  total_miles_walked 6 2 1 3 = 102 := by
  sorry

#eval total_miles_walked 6 2 1 3

end benjamin_weekly_miles_l1970_197049


namespace inclination_angle_range_l1970_197077

theorem inclination_angle_range (α : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0) →
  0 ≤ θ ∧ θ < π →
  (θ ∈ Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π) ↔ 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0 ∧ θ = Real.arctan (Real.sin α)) :=
by sorry

end inclination_angle_range_l1970_197077


namespace intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l1970_197037

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := x^2 - (a + 1)*x + a < 0

-- Theorems for the solution sets of the inequality
theorem solution_set_a_eq_1 : {x | inequality 1 x} = ∅ := by sorry

theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) : 
  {x | inequality a x} = {x | 1 < x ∧ x < a} := by sorry

theorem solution_set_a_lt_1 (a : ℝ) (h : a < 1) : 
  {x | inequality a x} = {x | a < x ∧ x < 1} := by sorry

end intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l1970_197037


namespace three_Z_five_equals_32_l1970_197086

/-- The Z operation as defined in the problem -/
def Z (a b : ℝ) : ℝ := b + 12 * a - a^2

/-- Theorem stating that 3 Z 5 equals 32 -/
theorem three_Z_five_equals_32 : Z 3 5 = 32 := by
  sorry

end three_Z_five_equals_32_l1970_197086


namespace total_arrangements_l1970_197093

-- Define the number of people
def total_people : ℕ := 5

-- Define the number of positions for person A
def positions_for_A : ℕ := 2

-- Define the number of positions for person B
def positions_for_B : ℕ := 3

-- Define the number of remaining people
def remaining_people : ℕ := total_people - 2

-- Theorem statement
theorem total_arrangements :
  (positions_for_A * positions_for_B * (Nat.factorial remaining_people)) = 36 := by
  sorry

end total_arrangements_l1970_197093


namespace parallel_vectors_m_value_l1970_197045

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -1 := by
sorry

end parallel_vectors_m_value_l1970_197045


namespace geometric_sequence_sum_l1970_197008

theorem geometric_sequence_sum (a₀ r : ℚ) (n : ℕ) (h₁ : a₀ = 1/3) (h₂ : r = 1/3) (h₃ : n = 10) :
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 29524/59049 := by
  sorry

end geometric_sequence_sum_l1970_197008


namespace total_days_2010_to_2015_l1970_197055

def is_leap_year (year : ℕ) : Bool :=
  year = 2012

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def years_in_period : List ℕ := [2010, 2011, 2012, 2013, 2014, 2015]

theorem total_days_2010_to_2015 :
  (years_in_period.map days_in_year).sum = 2191 := by
  sorry

end total_days_2010_to_2015_l1970_197055


namespace dot_product_OM_ON_l1970_197066

/-- Regular triangle OAB with side length 1 -/
structure RegularTriangle where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_regular : sorry
  side_length : sorry

/-- Points M and N divide AB into three equal parts -/
def divide_side (t : RegularTriangle) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Main theorem -/
theorem dot_product_OM_ON (t : RegularTriangle) : 
  let (M, N) := divide_side t
  let m := vec t.O M
  let n := vec t.O N
  dot_product m n = 1/6 := by
    sorry

end dot_product_OM_ON_l1970_197066


namespace normal_trip_time_l1970_197043

theorem normal_trip_time 
  (normal_distance : ℝ) 
  (additional_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : normal_distance = 150) 
  (h2 : additional_distance = 100) 
  (h3 : total_time = 5) :
  (normal_distance / ((normal_distance + additional_distance) / total_time)) = 3 := by
  sorry

end normal_trip_time_l1970_197043


namespace at_least_two_positive_l1970_197053

theorem at_least_two_positive (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c > 0) (h5 : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) := by
  sorry

end at_least_two_positive_l1970_197053


namespace liquid_volume_in_tin_l1970_197040

/-- The volume of liquid in a cylindrical tin with a conical cavity -/
theorem liquid_volume_in_tin (tin_diameter tin_height : ℝ) 
  (liquid_fill_ratio : ℝ) (cavity_height cavity_diameter : ℝ) : 
  tin_diameter = 10 →
  tin_height = 5 →
  liquid_fill_ratio = 2/3 →
  cavity_height = 2 →
  cavity_diameter = 4 →
  (liquid_fill_ratio * tin_height * π * (tin_diameter/2)^2 - 
   (1/3) * π * (cavity_diameter/2)^2 * cavity_height) = (242/3) * π := by
  sorry

#check liquid_volume_in_tin

end liquid_volume_in_tin_l1970_197040


namespace grape_juice_mixture_proof_l1970_197035

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture
    containing 20% grape juice results in a new mixture with 36% grape juice. -/
theorem grape_juice_mixture_proof :
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.20
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.36
  let initial_juice : ℝ := initial_mixture * initial_concentration
  let final_mixture : ℝ := initial_mixture + added_juice
  let final_juice : ℝ := initial_juice + added_juice
  final_juice / final_mixture = final_concentration :=
by sorry

end grape_juice_mixture_proof_l1970_197035


namespace max_value_problem_l1970_197005

theorem max_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3*x + 6*y < 108) :
  (x^2 * y * (108 - 3*x - 6*y)) ≤ 7776 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 6*y₀ < 108 ∧
    x₀^2 * y₀ * (108 - 3*x₀ - 6*y₀) = 7776 :=
by sorry

end max_value_problem_l1970_197005


namespace min_value_of_f_l1970_197060

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem min_value_of_f (m : ℝ) (h1 : -1 ≤ m) (h2 : m ≤ 1) :
  f m ≥ -4 ∧ ∃ m₀, -1 ≤ m₀ ∧ m₀ ≤ 1 ∧ f m₀ = -4 :=
sorry

end min_value_of_f_l1970_197060


namespace original_order_cost_l1970_197030

def original_tomatoes : ℝ := 0.99
def new_tomatoes : ℝ := 2.20
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_and_tip : ℝ := 8.00
def new_bill : ℝ := 35.00

theorem original_order_cost :
  let tomatoes_diff := new_tomatoes - original_tomatoes
  let lettuce_diff := new_lettuce - original_lettuce
  let celery_diff := new_celery - original_celery
  let total_diff := tomatoes_diff + lettuce_diff + celery_diff
  new_bill - delivery_and_tip - total_diff = 25 := by sorry

end original_order_cost_l1970_197030


namespace three_digit_multiples_of_seven_l1970_197018

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0) → 
  (∃ k, k = (Nat.floor (999 / 7) - Nat.ceil (100 / 7) + 1) ∧ k = 128) :=
by sorry

end three_digit_multiples_of_seven_l1970_197018


namespace russian_alphabet_symmetry_partition_l1970_197063

-- Define the set of Russian alphabet letters
inductive RussianLetter
| A | B | V | G | D | E | Zh | Z | I | K | L | M | N | O | P | R | S | T | U | F | Kh | Ts | Ch | Sh | Shch | Eh | Yu | Ya

-- Define symmetry types
inductive SymmetryType
| Vertical
| Horizontal
| Central
| All
| None

-- Define a function that assigns a symmetry type to each letter
def letterSymmetry : RussianLetter → SymmetryType
| RussianLetter.A => SymmetryType.Vertical
| RussianLetter.D => SymmetryType.Vertical
| RussianLetter.M => SymmetryType.Vertical
| RussianLetter.P => SymmetryType.Vertical
| RussianLetter.T => SymmetryType.Vertical
| RussianLetter.Sh => SymmetryType.Vertical
| RussianLetter.V => SymmetryType.Horizontal
| RussianLetter.E => SymmetryType.Horizontal
| RussianLetter.Z => SymmetryType.Horizontal
| RussianLetter.K => SymmetryType.Horizontal
| RussianLetter.S => SymmetryType.Horizontal
| RussianLetter.Eh => SymmetryType.Horizontal
| RussianLetter.Yu => SymmetryType.Horizontal
| RussianLetter.I => SymmetryType.Central
| RussianLetter.Zh => SymmetryType.All
| RussianLetter.N => SymmetryType.All
| RussianLetter.O => SymmetryType.All
| RussianLetter.F => SymmetryType.All
| RussianLetter.Kh => SymmetryType.All
| _ => SymmetryType.None

-- Define the five groups
def group1 := {l : RussianLetter | letterSymmetry l = SymmetryType.Vertical}
def group2 := {l : RussianLetter | letterSymmetry l = SymmetryType.Horizontal}
def group3 := {l : RussianLetter | letterSymmetry l = SymmetryType.Central}
def group4 := {l : RussianLetter | letterSymmetry l = SymmetryType.All}
def group5 := {l : RussianLetter | letterSymmetry l = SymmetryType.None}

-- Theorem: The groups form a partition of the Russian alphabet
theorem russian_alphabet_symmetry_partition :
  (∀ l : RussianLetter, l ∈ group1 ∨ l ∈ group2 ∨ l ∈ group3 ∨ l ∈ group4 ∨ l ∈ group5) ∧
  (group1 ∩ group2 = ∅) ∧ (group1 ∩ group3 = ∅) ∧ (group1 ∩ group4 = ∅) ∧ (group1 ∩ group5 = ∅) ∧
  (group2 ∩ group3 = ∅) ∧ (group2 ∩ group4 = ∅) ∧ (group2 ∩ group5 = ∅) ∧
  (group3 ∩ group4 = ∅) ∧ (group3 ∩ group5 = ∅) ∧
  (group4 ∩ group5 = ∅) :=
sorry

end russian_alphabet_symmetry_partition_l1970_197063


namespace max_remainder_theorem_l1970_197056

theorem max_remainder_theorem :
  (∀ n : ℕ, n < 120 → ∃ k : ℕ, 209 = k * n + 104 ∧ ∀ m : ℕ, m < n → 209 % m ≤ 104) ∧
  (∀ n : ℕ, n < 90 → ∃ k : ℕ, 209 = k * n + 69 ∧ ∀ m : ℕ, m < n → 209 % m ≤ 69) :=
by sorry

end max_remainder_theorem_l1970_197056


namespace jason_initial_cards_l1970_197088

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Benny bought from Jason -/
def cards_bought : ℕ := 2

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 1

/-- Theorem stating that Jason's initial number of Pokemon cards was 3 -/
theorem jason_initial_cards : initial_cards = 3 := by sorry

end jason_initial_cards_l1970_197088


namespace tangent_line_to_circle_l1970_197072

theorem tangent_line_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + (y - 3)^2 = 5 ∧ y = 2*x) →
  (a = -1 ∨ a = 4) := by
  sorry

end tangent_line_to_circle_l1970_197072


namespace binary_253_ones_minus_zeros_l1970_197016

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countTrue (l : List Bool) : ℕ :=
  l.foldl (fun acc b => if b then acc + 1 else acc) 0

/-- Count the number of false values in a list of booleans -/
def countFalse (l : List Bool) : ℕ :=
  l.length - countTrue l

theorem binary_253_ones_minus_zeros : 
  let binary := toBinary 253
  let ones := countTrue binary
  let zeros := countFalse binary
  ones - zeros = 6 := by sorry

end binary_253_ones_minus_zeros_l1970_197016


namespace set_A_characterization_l1970_197091

theorem set_A_characterization (A : Set ℕ) : 
  ({1} ∪ A = {1, 3, 5}) → (A = {1, 3, 5} ∨ A = {3, 5}) := by
  sorry

end set_A_characterization_l1970_197091


namespace sequence_value_at_50_l1970_197024

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + n + 1

theorem sequence_value_at_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 65 → f 50 = 257551 :=
by
  sorry

end sequence_value_at_50_l1970_197024


namespace coat_final_price_coat_price_is_81_l1970_197097

/-- The final price of a coat after discounts and tax -/
theorem coat_final_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (additional_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + tax_rate)
  final_price

/-- Proof that the final price of the coat is $81 -/
theorem coat_price_is_81 : 
  coat_final_price 100 0.2 5 0.08 = 81 := by
  sorry

end coat_final_price_coat_price_is_81_l1970_197097


namespace older_brother_pocket_money_l1970_197003

theorem older_brother_pocket_money
  (total_money : ℕ)
  (difference : ℕ)
  (h1 : total_money = 12000)
  (h2 : difference = 1000) :
  ∃ (younger older : ℕ),
    younger + older = total_money ∧
    older = younger + difference ∧
    older = 6500 := by
  sorry

end older_brother_pocket_money_l1970_197003


namespace equation_solution_l1970_197023

theorem equation_solution :
  ∀ x : ℝ, (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end equation_solution_l1970_197023


namespace intersection_of_A_and_B_l1970_197080

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l1970_197080


namespace complement_of_union_equals_four_l1970_197075

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {1, 2}
def Q : Set Nat := {2, 3}

theorem complement_of_union_equals_four : 
  (U \ (P ∪ Q)) = {4} := by sorry

end complement_of_union_equals_four_l1970_197075


namespace cafeteria_tile_problem_l1970_197014

theorem cafeteria_tile_problem :
  let current_tiles : ℕ := 630
  let current_area : ℕ := 18
  let new_tile_side : ℕ := 6
  let new_tiles : ℕ := 315
  (current_tiles * current_area = new_tiles * new_tile_side * new_tile_side) :=
by sorry

end cafeteria_tile_problem_l1970_197014


namespace percent_problem_l1970_197001

theorem percent_problem (x : ℝ) : (0.15 * 40 = 0.25 * x + 2) → x = 16 := by
  sorry

end percent_problem_l1970_197001


namespace binomial_coefficient_formula_l1970_197012

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) : 
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) := by
  sorry

end binomial_coefficient_formula_l1970_197012


namespace subsets_and_sum_of_M_l1970_197004

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (fun s => s.sum id) = 55 * 2^9 := by
  sorry

end subsets_and_sum_of_M_l1970_197004


namespace solve_equation_l1970_197034

theorem solve_equation : ∃ x : ℝ, (4 / 7) * (1 / 5) * x = 2 ∧ x = 35 / 2 := by
  sorry

end solve_equation_l1970_197034


namespace triangle_properties_l1970_197029

/-- Given a triangle ABC with the following properties:
    - B has coordinates (1, -2)
    - The median CM on side AB has equation 2x - y + 1 = 0
    - The angle bisector of ∠BAC has equation x + 7y - 12 = 0
    Prove that:
    1. A has coordinates (-2, 2)
    2. The equation of line AC is 3x - 4y + 14 = 0
-/
theorem triangle_properties (B : ℝ × ℝ) (median_CM : ℝ → ℝ → ℝ) (angle_bisector : ℝ → ℝ → ℝ) 
  (hB : B = (1, -2))
  (hmedian : ∀ x y, median_CM x y = 2 * x - y + 1)
  (hbisector : ∀ x y, angle_bisector x y = x + 7 * y - 12) :
  ∃ (A : ℝ × ℝ) (line_AC : ℝ → ℝ → ℝ),
    A = (-2, 2) ∧ 
    (∀ x y, line_AC x y = 3 * x - 4 * y + 14) := by
  sorry

end triangle_properties_l1970_197029


namespace angle_x_value_l1970_197011

/-- Given a configuration where AB and CD are straight lines, with specific angle measurements, prove that angle x equals 35 degrees. -/
theorem angle_x_value (AXB CYX XYB : ℝ) (h1 : AXB = 150) (h2 : CYX = 130) (h3 : XYB = 55) : ∃ x : ℝ, x = 35 := by
  sorry

end angle_x_value_l1970_197011


namespace max_value_of_x_l1970_197092

theorem max_value_of_x (x : ℝ) : 
  (((4 * x - 16) / (3 * x - 4)) ^ 2 + (4 * x - 16) / (3 * x - 4) = 18) →
  x ≤ (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
by sorry

end max_value_of_x_l1970_197092


namespace rhombus_perimeter_l1970_197094

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

end rhombus_perimeter_l1970_197094


namespace sum_even_coefficients_is_seven_l1970_197062

/-- Given a polynomial equation, prove that the sum of even-indexed coefficients (excluding a₀) is 7 -/
theorem sum_even_coefficients_is_seven (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^8 = a*x^12 + a₁*x^11 + a₂*x^10 + a₃*x^9 + a₄*x^8 + 
    a₅*x^7 + a₆*x^6 + a₇*x^5 + a₈*x^4 + a₉*x^3 + a₁₀*x^2 + a₁₁*x + a₁₂) →
  a = 1 →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 7 := by
sorry


end sum_even_coefficients_is_seven_l1970_197062


namespace union_M_N_complement_intersection_M_N_l1970_197084

-- Define the universal set U
def U : Set ℝ := {x | -6 ≤ x ∧ x ≤ 5}

-- Define set M
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for ∁U(M ∩ N)
theorem complement_intersection_M_N : 
  (M ∩ N)ᶜ = {x ∈ U | x ≤ 0 ∨ 2 ≤ x} := by sorry

end union_M_N_complement_intersection_M_N_l1970_197084


namespace intersection_implies_sum_l1970_197087

/-- Given two graphs that intersect at (3,4) and (7,2), prove that a+c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - (a + 1)| + b = |x - (c - 1)| + (d - 1) → x = 3 ∨ x = 7) →
  -|3 - (a + 1)| + b = |3 - (c - 1)| + (d - 1) →
  -|7 - (a + 1)| + b = |7 - (c - 1)| + (d - 1) →
  -|3 - (a + 1)| + b = 4 →
  -|7 - (a + 1)| + b = 2 →
  |3 - (c - 1)| + (d - 1) = 4 →
  |7 - (c - 1)| + (d - 1) = 2 →
  a + c = 10 := by
sorry

end intersection_implies_sum_l1970_197087


namespace hyperbola_equation_l1970_197009

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = Real.sqrt 3)
  (θ : ℝ) (h4 : Real.tan θ = Real.sqrt 21 / 2)
  (P Q : ℝ × ℝ) (F2 : ℝ × ℝ) (h5 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h6 : Q.1 = 0) (h7 : (Q.2 - F2.2) / (Q.1 - F2.1) = Real.tan θ)
  (h8 : dist P Q / dist P F2 = 1/2) :
  ∃ (k : ℝ), ∀ (x y : ℝ), 3 * x^2 - y^2 = k :=
by sorry

end hyperbola_equation_l1970_197009


namespace volume_to_surface_area_ratio_l1970_197069

/-- A convex polyhedron inscribed around a sphere -/
structure InscribedPolyhedron where
  -- The radius of the inscribed sphere
  r : ℝ
  -- The surface area of the polyhedron
  surface_area : ℝ
  -- The volume of the polyhedron
  volume : ℝ
  -- Assumption that the polyhedron is inscribed around the sphere
  inscribed : True

/-- 
Theorem: For any convex polyhedron inscribed around a sphere,
the ratio of its volume to its surface area is equal to r/3,
where r is the radius of the inscribed sphere.
-/
theorem volume_to_surface_area_ratio (P : InscribedPolyhedron) :
  P.volume / P.surface_area = P.r / 3 := by
  sorry

end volume_to_surface_area_ratio_l1970_197069


namespace unique_tangent_circle_l1970_197061

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Configuration of three unit circles each tangent to the other two -/
def unit_circle_configuration (c1 c2 c3 : Circle) : Prop :=
  c1.radius = 1 ∧ c2.radius = 1 ∧ c3.radius = 1 ∧
  are_tangent c1 c2 ∧ are_tangent c2 c3 ∧ are_tangent c3 c1

/-- A circle of radius 2 tangent to all three unit circles -/
def tangent_circle (c : Circle) (c1 c2 c3 : Circle) : Prop :=
  c.radius = 2 ∧ are_tangent c c1 ∧ are_tangent c c2 ∧ are_tangent c c3

theorem unique_tangent_circle (c1 c2 c3 : Circle) :
  unit_circle_configuration c1 c2 c3 →
  ∃! c : Circle, tangent_circle c c1 c2 c3 := by
  sorry

end unique_tangent_circle_l1970_197061


namespace system_of_equations_sum_l1970_197082

theorem system_of_equations_sum (x y z : ℝ) 
  (eq1 : y + z = 16 - 4*x)
  (eq2 : x + z = -18 - 4*y)
  (eq3 : x + y = 13 - 4*z) :
  2*x + 2*y + 2*z = 11/3 := by
sorry

end system_of_equations_sum_l1970_197082


namespace polynomial_always_positive_l1970_197032

theorem polynomial_always_positive (x y : ℝ) : x^2 + y^2 - 2*x - 4*y + 16 > 0 := by
  sorry

end polynomial_always_positive_l1970_197032


namespace tony_lego_purchase_l1970_197078

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_price : ℕ
  sword_price : ℕ
  dough_price : ℕ
  sword_count : ℕ
  dough_count : ℕ
  total_paid : ℕ

/-- Calculates the number of Lego sets bought -/
def lego_sets_bought (purchase : ToyPurchase) : ℕ :=
  (purchase.total_paid - purchase.sword_price * purchase.sword_count - purchase.dough_price * purchase.dough_count) / purchase.lego_price

/-- Theorem stating that Tony bought 3 sets of Lego blocks -/
theorem tony_lego_purchase : 
  ∀ (purchase : ToyPurchase), 
  purchase.lego_price = 250 ∧ 
  purchase.sword_price = 120 ∧ 
  purchase.dough_price = 35 ∧ 
  purchase.sword_count = 7 ∧ 
  purchase.dough_count = 10 ∧ 
  purchase.total_paid = 1940 → 
  lego_sets_bought purchase = 3 := by
  sorry

end tony_lego_purchase_l1970_197078


namespace imaginary_part_of_complex_number_l1970_197039

theorem imaginary_part_of_complex_number (b : ℝ) :
  let z : ℂ := 2 + b * Complex.I
  (Complex.abs z = 2 * Real.sqrt 2) → (b = 2 ∨ b = -2) :=
by sorry

end imaginary_part_of_complex_number_l1970_197039


namespace extreme_points_inequality_l1970_197054

/-- The function f(x) = x - a/x - 2ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - 2 * Real.log x

/-- Predicate to check if x is an extreme point of f -/
def is_extreme_point (a : ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f a y ≠ f a x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  is_extreme_point a x₁ →
  is_extreme_point a x₂ →
  f a x₂ < x₂ - 1 := by
  sorry

end extreme_points_inequality_l1970_197054


namespace larger_number_problem_l1970_197079

theorem larger_number_problem (a b : ℝ) : 
  a + b = 40 → a - b = 10 → a > b → a = 25 := by
sorry

end larger_number_problem_l1970_197079


namespace tan_two_pi_fifth_plus_theta_l1970_197057

theorem tan_two_pi_fifth_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
  sorry

end tan_two_pi_fifth_plus_theta_l1970_197057


namespace and_sufficient_not_necessary_for_or_l1970_197033

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end and_sufficient_not_necessary_for_or_l1970_197033


namespace corners_count_is_even_l1970_197050

/-- A corner is a shape on a grid paper -/
structure Corner where
  position : ℤ × ℤ

/-- A rectangle is a 1x4 shape on a grid paper -/
structure Rectangle where
  position : ℤ × ℤ

/-- A centrally symmetric figure on a grid paper -/
structure CentrallySymmetricFigure where
  corners : List Corner
  rectangles : List Rectangle
  is_centrally_symmetric : Bool

/-- The theorem states that in a centrally symmetric figure composed of corners and 1x4 rectangles, 
    the number of corners must be even -/
theorem corners_count_is_even (figure : CentrallySymmetricFigure) 
  (h : figure.is_centrally_symmetric = true) : 
  Even (figure.corners.length) := by
  sorry

end corners_count_is_even_l1970_197050


namespace parallel_vectors_condition_l1970_197017

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The statement that m = 1 is a necessary but not sufficient condition for
    vectors (m, 1) and (1, m) to be parallel -/
theorem parallel_vectors_condition :
  ∃ m : ℝ, (m = 1 → are_parallel (m, 1) (1, m)) ∧
           ¬(are_parallel (m, 1) (1, m) → m = 1) :=
sorry

end parallel_vectors_condition_l1970_197017


namespace lisa_notebook_savings_l1970_197002

/-- Calculates the savings when buying notebooks with discounts -/
def notebook_savings (
  quantity : ℕ
  ) (original_price : ℚ
  ) (discount_rate : ℚ
  ) (bulk_discount : ℚ
  ) (bulk_threshold : ℕ
  ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_discount := quantity * discounted_price
  let final_total := 
    if quantity > bulk_threshold
    then total_with_discount - bulk_discount
    else total_with_discount
  total_without_discount - final_total

/-- Theorem stating the savings for Lisa's notebook purchase -/
theorem lisa_notebook_savings :
  notebook_savings 8 3 (30/100) 5 7 = 61/5 := by
  sorry

end lisa_notebook_savings_l1970_197002


namespace sequence_sum_l1970_197025

/-- Given a geometric sequence {a_n} and an arithmetic sequence {b_n}, prove that
    b_3 + b_11 = 6 under the given conditions. -/
theorem sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 2 * a 3 * a 4 = 27 / 64 →   -- product condition
  b 7 = a 5 →                   -- relation between sequences
  (∃ d, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 3 + b 11 = 6 := by
sorry

end sequence_sum_l1970_197025


namespace second_book_length_is_100_l1970_197059

/-- The length of Yasna's first book in pages -/
def first_book_length : ℕ := 180

/-- The number of pages Yasna reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The total number of pages Yasna reads in two weeks -/
def total_pages : ℕ := pages_per_day * days_in_two_weeks

/-- The length of Yasna's second book in pages -/
def second_book_length : ℕ := total_pages - first_book_length

theorem second_book_length_is_100 : second_book_length = 100 := by
  sorry

end second_book_length_is_100_l1970_197059


namespace dividend_percentage_calculation_l1970_197021

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 840.0000000000001) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 7 := by
sorry

end dividend_percentage_calculation_l1970_197021


namespace acid_dilution_l1970_197010

/-- Given an initial acid solution with concentration p% and volume p ounces,
    adding y ounces of water results in a (p-15)% acid solution.
    This theorem proves that y = 15p / (p-15) when p > 30. -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h : p > 30) :
  (p * p / 100 = (p - 15) / 100 * (p + y)) → y = 15 * p / (p - 15) := by
  sorry


end acid_dilution_l1970_197010


namespace cost_price_calculation_l1970_197015

/-- Proves that the cost price is 17500 given the selling price, discount rate, and profit rate --/
theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
  sorry

#check cost_price_calculation

end cost_price_calculation_l1970_197015


namespace exponential_inequality_solution_set_l1970_197095

theorem exponential_inequality_solution_set :
  {x : ℝ | (4 : ℝ)^(8 - x) > (4 : ℝ)^(-2 * x)} = {x : ℝ | x > -8} := by sorry

end exponential_inequality_solution_set_l1970_197095


namespace binomial_coefficient_is_integer_l1970_197083

theorem binomial_coefficient_is_integer (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, (m.choose n) = k := by sorry

end binomial_coefficient_is_integer_l1970_197083


namespace valid_a_value_l1970_197070

-- Define the linear equation
def linear_equation (a x : ℝ) : Prop := (a - 1) * x - 6 = 0

-- State the theorem
theorem valid_a_value : ∃ (a : ℝ), a ≠ 1 ∧ ∀ (x : ℝ), linear_equation a x → True :=
by
  sorry

end valid_a_value_l1970_197070


namespace system_solution_ratio_l1970_197085

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 5 * y = a →
  10 * y - 15 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 15 := by
sorry

end system_solution_ratio_l1970_197085


namespace min_sphere_surface_area_for_pyramid_l1970_197064

/-- Minimum surface area of a sphere containing a specific triangular pyramid -/
theorem min_sphere_surface_area_for_pyramid (V : ℝ) (h : ℝ) (angle : ℝ) : 
  V = 8 * Real.sqrt 3 →
  h = 4 →
  angle = π / 3 →
  ∃ (S : ℝ), S = 48 * π ∧ 
    ∀ (S' : ℝ), (∃ (r : ℝ), S' = 4 * π * r^2 ∧ 
      ∃ (a b c : ℝ), 
        a^2 + (h/2)^2 ≤ r^2 ∧
        b^2 + (h/2)^2 ≤ r^2 ∧
        c^2 + h^2 ≤ r^2 ∧
        (1/3) * (1/2) * a * b * Real.sin angle * h = V) → 
    S ≤ S' :=
sorry

end min_sphere_surface_area_for_pyramid_l1970_197064


namespace p_sufficient_not_necessary_for_q_l1970_197027

-- Define propositions p and q
def p (a b : ℝ) : Prop := a > 0 ∧ 0 > b

def q (a b : ℝ) : Prop := |a + b| < |a| + |b|

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, p a b → q a b) ∧
  ¬(∀ a b : ℝ, q a b → p a b) :=
sorry

end p_sufficient_not_necessary_for_q_l1970_197027


namespace inconsistent_equation_l1970_197028

theorem inconsistent_equation : ¬ (3 * (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005) := by
  sorry

end inconsistent_equation_l1970_197028


namespace absolute_value_quadratic_equivalence_l1970_197026

theorem absolute_value_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = 8 ∧ c = 7 := by
sorry

end absolute_value_quadratic_equivalence_l1970_197026


namespace visibility_condition_l1970_197019

/-- The curve C: y = 2x^2 -/
def C (x : ℝ) : ℝ := 2 * x^2

/-- Point A -/
def A : ℝ × ℝ := (0, -2)

/-- Point B -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- A point (x, y) is above the curve C -/
def is_above_curve (x y : ℝ) : Prop := y > C x

/-- A point (x, y) is on or below the line passing through two points -/
def is_on_or_below_line (x1 y1 x2 y2 x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) ≤ (y2 - y1) * (x - x1)

/-- B is visible from A without being obstructed by C -/
def is_visible (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 →
    is_above_curve x ((a + 2) / 3 * x - 2)

theorem visibility_condition (a : ℝ) :
  is_visible a ↔ a < 10 := by sorry

end visibility_condition_l1970_197019


namespace min_saplings_needed_l1970_197006

theorem min_saplings_needed (road_length : ℕ) (tree_spacing : ℕ) : road_length = 1000 → tree_spacing = 100 → 
  (road_length / tree_spacing + 1) * 2 = 22 := by
  sorry

end min_saplings_needed_l1970_197006


namespace opposite_sides_iff_m_range_l1970_197096

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation 3x - y + m = 0 -/
def lineEquation (p : Point) (m : ℝ) : ℝ := 3 * p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def oppositeSides (p1 p2 : Point) (m : ℝ) : Prop :=
  lineEquation p1 m * lineEquation p2 m < 0

/-- The theorem stating the equivalence between the points being on opposite sides and the range of m -/
theorem opposite_sides_iff_m_range (m : ℝ) :
  oppositeSides (Point.mk 1 2) (Point.mk 1 1) m ↔ -2 < m ∧ m < -1 := by
  sorry

end opposite_sides_iff_m_range_l1970_197096


namespace rectangle_dimensions_l1970_197065

/-- The dimensions of a rectangle satisfying specific conditions -/
theorem rectangle_dimensions :
  ∀ x y : ℝ,
  x > 0 ∧ y > 0 →  -- Ensure positive dimensions
  y = 2 * x →      -- Length is twice the width
  2 * (x + y) = 2 * (x * y) →  -- Perimeter is twice the area
  (x, y) = (3/2, 3) := by
sorry

end rectangle_dimensions_l1970_197065


namespace abs_equation_solution_difference_l1970_197042

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 30) := by
sorry

end abs_equation_solution_difference_l1970_197042


namespace sqrt2_irrational_l1970_197099

theorem sqrt2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 2 := by
  sorry

end sqrt2_irrational_l1970_197099


namespace arithmetic_sequence_sum_l1970_197044

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms starting from the third term is 250 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 250

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 2 + a 8 = 100 := by
  sorry

end arithmetic_sequence_sum_l1970_197044


namespace clothing_pricing_solution_l1970_197038

/-- Represents the pricing strategy for a piece of clothing --/
structure ClothingPricing where
  markedPrice : ℝ
  costPrice : ℝ

/-- Defines the conditions for the clothing pricing problem --/
def validPricing (p : ClothingPricing) : Prop :=
  (0.5 * p.markedPrice + 20 = p.costPrice) ∧ 
  (0.8 * p.markedPrice - 40 = p.costPrice)

/-- Theorem stating the unique solution to the clothing pricing problem --/
theorem clothing_pricing_solution :
  ∃! p : ClothingPricing, validPricing p ∧ p.markedPrice = 200 ∧ p.costPrice = 120 := by
  sorry


end clothing_pricing_solution_l1970_197038


namespace excircle_incircle_similarity_l1970_197081

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle defined by its center and a point on the circumference -/
structure Circle :=
  (center : Point) (point : Point)

/-- Defines an excircle of a triangle -/
def excircle (T : Triangle) (vertex : Point) : Circle :=
  sorry

/-- Defines the incircle of a triangle -/
def incircle (T : Triangle) : Circle :=
  sorry

/-- Defines the circumcircle of a triangle -/
def circumcircle (T : Triangle) : Circle :=
  sorry

/-- Defines the point where a circle touches a line segment -/
def touchPoint (C : Circle) (A B : Point) : Point :=
  sorry

/-- Defines the intersection points of two circles -/
def circleIntersection (C1 C2 : Circle) : Set Point :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

theorem excircle_incircle_similarity
  (ABC : Triangle)
  (A' : Point) (B' : Point) (C' : Point)
  (C1 : Point) (A1 : Point) (B1 : Point) :
  A' = touchPoint (excircle ABC ABC.A) ABC.B ABC.C →
  B' = touchPoint (excircle ABC ABC.B) ABC.C ABC.A →
  C' = touchPoint (excircle ABC ABC.C) ABC.A ABC.B →
  C1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', B', C⟩) →
  A1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨ABC.A, B', C'⟩) →
  B1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', ABC.B, C'⟩) →
  let incirclePoints := Triangle.mk
    (touchPoint (incircle ABC) ABC.B ABC.C)
    (touchPoint (incircle ABC) ABC.C ABC.A)
    (touchPoint (incircle ABC) ABC.A ABC.B)
  areSimilar ⟨A1, B1, C1⟩ incirclePoints :=
by
  sorry

end excircle_incircle_similarity_l1970_197081
