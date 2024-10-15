import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l3722_372241

theorem expression_simplification : 120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3722_372241


namespace NUMINAMATH_CALUDE_cube_edge_length_proof_l3722_372286

-- Define the vessel dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15
def water_level_rise : ℝ := 3.3333333333333335

-- Define the cube's edge length
def cube_edge_length : ℝ := 10

-- Theorem statement
theorem cube_edge_length_proof :
  let vessel_base_area := vessel_length * vessel_width
  let water_volume_displaced := vessel_base_area * water_level_rise
  water_volume_displaced = cube_edge_length ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_proof_l3722_372286


namespace NUMINAMATH_CALUDE_chenny_initial_candies_l3722_372264

/-- The number of friends Chenny has -/
def num_friends : ℕ := 7

/-- The number of candies each friend should receive -/
def candies_per_friend : ℕ := 2

/-- The number of additional candies Chenny needs to buy -/
def additional_candies : ℕ := 4

/-- Chenny's initial number of candies -/
def initial_candies : ℕ := num_friends * candies_per_friend - additional_candies

theorem chenny_initial_candies : initial_candies = 10 := by
  sorry

end NUMINAMATH_CALUDE_chenny_initial_candies_l3722_372264


namespace NUMINAMATH_CALUDE_square_of_binomial_l3722_372297

theorem square_of_binomial (x : ℝ) : (7 - Real.sqrt (x^2 - 33))^2 = x^2 - 14 * Real.sqrt (x^2 - 33) + 16 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3722_372297


namespace NUMINAMATH_CALUDE_total_entertainment_hours_l3722_372295

/-- Represents the hours spent on an activity for each day of the week -/
structure WeeklyHours :=
  (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ)
  (friday : ℕ) (saturday : ℕ) (sunday : ℕ)

/-- Calculates the total hours spent on an activity throughout the week -/
def totalHours (hours : WeeklyHours) : ℕ :=
  hours.monday + hours.tuesday + hours.wednesday + hours.thursday +
  hours.friday + hours.saturday + hours.sunday

/-- Haley's TV watching hours -/
def tvHours : WeeklyHours :=
  { monday := 0, tuesday := 2, wednesday := 0, thursday := 4,
    friday := 0, saturday := 6, sunday := 3 }

/-- Haley's video game playing hours -/
def gameHours : WeeklyHours :=
  { monday := 3, tuesday := 0, wednesday := 5, thursday := 0,
    friday := 1, saturday := 0, sunday := 0 }

theorem total_entertainment_hours :
  totalHours tvHours + totalHours gameHours = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_entertainment_hours_l3722_372295


namespace NUMINAMATH_CALUDE_remaining_numbers_count_l3722_372208

theorem remaining_numbers_count (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 8 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 14 →
  (total - subset = 2 ∧ 
   (total * total_avg - subset * subset_avg) / (total - subset) = remaining_avg) :=
by sorry

end NUMINAMATH_CALUDE_remaining_numbers_count_l3722_372208


namespace NUMINAMATH_CALUDE_age_problem_l3722_372270

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 52
  Prove that b is 20 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 52) :
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3722_372270


namespace NUMINAMATH_CALUDE_existence_of_m_l3722_372273

theorem existence_of_m (a b : ℝ) (h : a > b) : ∃ m : ℝ, a * m < b * m := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l3722_372273


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_l3722_372224

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ := by
  -- Define the number of sides of a decagon
  let n : ℕ := 10

  -- Define the sum of interior angles formula
  let sum_of_interior_angles (sides : ℕ) : ℝ := (sides - 2) * 180

  -- Calculate the sum of interior angles for a decagon
  let total_angle_sum : ℝ := sum_of_interior_angles n

  -- Calculate the measure of one interior angle
  let interior_angle : ℝ := total_angle_sum / n

  -- Prove that the interior angle is 144 degrees
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_l3722_372224


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l3722_372205

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l3722_372205


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3722_372242

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (a b : Line)
  (h1 : perp_plane α β)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane b β) :
  perp_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3722_372242


namespace NUMINAMATH_CALUDE_equation_solution_sum_l3722_372212

theorem equation_solution_sum : ∃ x₁ x₂ : ℝ, 
  (6 * x₁) / 30 = 7 / x₁ ∧
  (6 * x₂) / 30 = 7 / x₂ ∧
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l3722_372212


namespace NUMINAMATH_CALUDE_primeDivisorsOf50FactorialIs15_l3722_372263

/-- The number of prime divisors of 50! -/
def primeDivisorsOf50Factorial : ℕ :=
  (List.range 51).filter (fun n => n.Prime && n > 1) |>.length

/-- Theorem: The number of prime divisors of 50! is 15 -/
theorem primeDivisorsOf50FactorialIs15 : primeDivisorsOf50Factorial = 15 := by
  sorry

end NUMINAMATH_CALUDE_primeDivisorsOf50FactorialIs15_l3722_372263


namespace NUMINAMATH_CALUDE_multiply_106_94_l3722_372292

theorem multiply_106_94 : 106 * 94 = 9964 := by
  sorry

end NUMINAMATH_CALUDE_multiply_106_94_l3722_372292


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_binary_l3722_372272

theorem sum_of_gcd_and_binary : ∃ (a b : ℕ),
  (Nat.gcd 98 63 = a) ∧
  (((1 : ℕ) * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = b) ∧
  (a + b = 58) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_binary_l3722_372272


namespace NUMINAMATH_CALUDE_subset_of_square_eq_self_l3722_372276

theorem subset_of_square_eq_self : {1} ⊆ {x : ℝ | x^2 = x} := by sorry

end NUMINAMATH_CALUDE_subset_of_square_eq_self_l3722_372276


namespace NUMINAMATH_CALUDE_werewolf_victims_l3722_372215

/-- Given a village with a certain population, a vampire's weekly victim count, 
    and a time period, calculate the werewolf's weekly victim count. -/
theorem werewolf_victims (village_population : ℕ) (vampire_victims_per_week : ℕ) (weeks : ℕ) 
  (h1 : village_population = 72)
  (h2 : vampire_victims_per_week = 3)
  (h3 : weeks = 9) :
  ∃ (werewolf_victims_per_week : ℕ), 
    werewolf_victims_per_week * weeks + vampire_victims_per_week * weeks = village_population ∧ 
    werewolf_victims_per_week = 5 :=
by sorry

end NUMINAMATH_CALUDE_werewolf_victims_l3722_372215


namespace NUMINAMATH_CALUDE_g_of_3_eq_3_l3722_372254

/-- The function g is defined as g(x) = x^2 - 2x for all real x. -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- Theorem: The value of g(3) is 3. -/
theorem g_of_3_eq_3 : g 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_3_l3722_372254


namespace NUMINAMATH_CALUDE_union_complement_equals_l3722_372231

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {2, 5}

-- Theorem statement
theorem union_complement_equals : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equals_l3722_372231


namespace NUMINAMATH_CALUDE_marble_difference_l3722_372268

theorem marble_difference (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ)
  (h_total : total = 19)
  (h_yellow : yellow = 5)
  (h_blue_ratio : blue_ratio = 3)
  (h_red_ratio : red_ratio = 4) :
  let remaining : ℕ := total - yellow
  let share : ℕ := remaining / (blue_ratio + red_ratio)
  let red : ℕ := red_ratio * share
  red - yellow = 3 := by sorry

end NUMINAMATH_CALUDE_marble_difference_l3722_372268


namespace NUMINAMATH_CALUDE_square_roots_problem_l3722_372278

theorem square_roots_problem (a m : ℝ) : 
  ((2 - m)^2 = a ∧ (2*m + 1)^2 = a) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3722_372278


namespace NUMINAMATH_CALUDE_equation_condition_l3722_372275

theorem equation_condition (a b c : ℕ+) (hb : b < 12) (hc : c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l3722_372275


namespace NUMINAMATH_CALUDE_initial_bucket_capacity_is_5_l3722_372296

/-- The capacity of the initially filled bucket -/
def initial_bucket_capacity : ℝ := 5

/-- The capacity of the small bucket -/
def small_bucket_capacity : ℝ := 3

/-- The capacity of the large bucket -/
def large_bucket_capacity : ℝ := 6

/-- The amount of additional water the large bucket can hold -/
def additional_capacity : ℝ := 4

theorem initial_bucket_capacity_is_5 :
  initial_bucket_capacity = small_bucket_capacity + (large_bucket_capacity - additional_capacity) :=
by
  sorry

#check initial_bucket_capacity_is_5

end NUMINAMATH_CALUDE_initial_bucket_capacity_is_5_l3722_372296


namespace NUMINAMATH_CALUDE_no_base_for_172_four_digit_odd_final_l3722_372238

theorem no_base_for_172_four_digit_odd_final (b : ℕ) : ¬ (
  (b ^ 3 ≤ 172 ∧ 172 < b ^ 4) ∧  -- four-digit number condition
  (172 % b % 2 = 1)              -- odd final digit condition
) := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_172_four_digit_odd_final_l3722_372238


namespace NUMINAMATH_CALUDE_pauls_savings_l3722_372257

/-- Paul's initial savings in dollars -/
def initial_savings : ℕ := sorry

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Additional money Paul receives in dollars -/
def additional_money : ℕ := 7

theorem pauls_savings :
  initial_savings = 3 ∧
  initial_savings + additional_money = num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_pauls_savings_l3722_372257


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3722_372293

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3722_372293


namespace NUMINAMATH_CALUDE_winter_clothes_cost_theorem_l3722_372210

/-- Represents the cost calculation for winter clothes with a discount --/
def winter_clothes_cost (total_children : ℕ) (toddlers : ℕ) 
  (toddler_cost school_cost preteen_cost teen_cost : ℕ) 
  (discount_percent : ℕ) : ℕ :=
  let school_age := 2 * toddlers
  let preteens := school_age / 2
  let teens := 4 * toddlers + toddlers
  let total_cost := toddler_cost * toddlers + 
                    school_cost * school_age + 
                    preteen_cost * preteens + 
                    teen_cost * teens
  let discount := preteen_cost * preteens * discount_percent / 100
  total_cost - discount

/-- Theorem stating the total cost of winter clothes after discount --/
theorem winter_clothes_cost_theorem :
  winter_clothes_cost 60 6 35 45 55 65 30 = 2931 := by
  sorry

#eval winter_clothes_cost 60 6 35 45 55 65 30

end NUMINAMATH_CALUDE_winter_clothes_cost_theorem_l3722_372210


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3722_372259

theorem intersection_point_of_lines (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 1) ↔ (x = 25/31 ∧ y = -22/31) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3722_372259


namespace NUMINAMATH_CALUDE_p_investment_l3722_372265

/-- Given that Q invested 15000 and the profit is divided in the ratio 5:1, prove that P's investment is 75000 --/
theorem p_investment (q_investment : ℕ) (profit_ratio_p profit_ratio_q : ℕ) :
  q_investment = 15000 →
  profit_ratio_p = 5 →
  profit_ratio_q = 1 →
  profit_ratio_p * q_investment = profit_ratio_q * 75000 :=
by sorry

end NUMINAMATH_CALUDE_p_investment_l3722_372265


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_sqrt_l3722_372204

theorem sqrt_plus_reciprocal_sqrt (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 98) :
  Real.sqrt x + 1 / Real.sqrt x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_sqrt_l3722_372204


namespace NUMINAMATH_CALUDE_inequality_chain_l3722_372200

theorem inequality_chain (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3722_372200


namespace NUMINAMATH_CALUDE_observation_mean_invariance_l3722_372220

theorem observation_mean_invariance (n : ℕ) (h : n > 0) :
  let original_mean : ℚ := 200
  let decrement : ℚ := 6
  let new_mean : ℚ := 194
  n * original_mean - n * decrement = n * new_mean :=
by
  sorry

end NUMINAMATH_CALUDE_observation_mean_invariance_l3722_372220


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3722_372277

/-- A rectangular solid with prime edge lengths and volume 385 has surface area 334 -/
theorem rectangular_solid_surface_area : 
  ∀ (l w h : ℕ), 
    Prime l → Prime w → Prime h →
    l * w * h = 385 →
    2 * (l * w + l * h + w * h) = 334 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3722_372277


namespace NUMINAMATH_CALUDE_square_area_quadrupled_l3722_372226

theorem square_area_quadrupled (a : ℝ) (h : a > 0) :
  (2 * a)^2 = 4 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_quadrupled_l3722_372226


namespace NUMINAMATH_CALUDE_cctv_systematic_sampling_group_size_l3722_372290

/-- Calculates the group size for systematic sampling -/
def systematicSamplingGroupSize (totalViewers : ℕ) (selectedViewers : ℕ) : ℕ :=
  totalViewers / selectedViewers

/-- Theorem: The group size for selecting 10 lucky viewers from 10000 viewers using systematic sampling is 1000 -/
theorem cctv_systematic_sampling_group_size :
  systematicSamplingGroupSize 10000 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cctv_systematic_sampling_group_size_l3722_372290


namespace NUMINAMATH_CALUDE_bal_puzzle_l3722_372225

/-- Represents the possible meanings of the word "bal" -/
inductive BalMeaning
  | Yes
  | No

/-- Represents the possible types of inhabitants -/
inductive InhabitantType
  | Human
  | Zombie

/-- Represents the response to a yes/no question -/
def Response := BalMeaning

/-- Models the behavior of an inhabitant based on their type -/
def inhabitantBehavior (t : InhabitantType) (actual : BalMeaning) (response : Response) : Prop :=
  match t with
  | InhabitantType.Human => response = actual
  | InhabitantType.Zombie => response ≠ actual

/-- The main theorem capturing the essence of the problem -/
theorem bal_puzzle (response : Response) :
  (∀ meaning : BalMeaning, ∃ t : InhabitantType, inhabitantBehavior t meaning response) ∧
  (∃! t : InhabitantType, ∀ meaning : BalMeaning, inhabitantBehavior t meaning response) :=
by sorry

end NUMINAMATH_CALUDE_bal_puzzle_l3722_372225


namespace NUMINAMATH_CALUDE_chocolate_candy_difference_l3722_372203

-- Define the cost of chocolate and candy bar
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 2

-- Theorem statement
theorem chocolate_candy_difference :
  chocolate_cost - candy_bar_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_candy_difference_l3722_372203


namespace NUMINAMATH_CALUDE_equation_solutions_l3722_372209

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = -9 ∧ x₁^2 + 12*x₁ + 27 = 0 ∧ x₂^2 + 12*x₂ + 27 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-5 + Real.sqrt 10) / 3 ∧ x₂ = (-5 - Real.sqrt 10) / 3 ∧
    3*x₁^2 + 10*x₁ + 5 = 0 ∧ 3*x₂^2 + 10*x₂ + 5 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4/3 ∧ x₂ = 2/3 ∧ (3*x₁ + 1)^2 - 9 = 0 ∧ (3*x₂ + 1)^2 - 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3722_372209


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l3722_372240

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel 
  (l : Line3D) (α β : Plane3D) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular l α) 
  (h_perp_β : perpendicular l β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l3722_372240


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l3722_372216

/-- Represents the structure of the sculpture --/
structure Sculpture :=
  (num_cubes : ℕ)
  (edge_length : ℝ)
  (top_layer : ℕ)
  (middle_layer : ℕ)
  (bottom_layer : ℕ)

/-- Calculates the exposed surface area of the sculpture --/
def exposed_surface_area (s : Sculpture) : ℝ :=
  let top_area := s.top_layer * (5 * s.edge_length^2 + s.edge_length^2)
  let middle_area := s.middle_layer * s.edge_length^2 + 8 * s.edge_length^2
  let bottom_area := s.bottom_layer * s.edge_length^2
  top_area + middle_area + bottom_area

/-- The main theorem to be proved --/
theorem sculpture_surface_area :
  ∀ s : Sculpture,
    s.num_cubes = 14 ∧
    s.edge_length = 1 ∧
    s.top_layer = 1 ∧
    s.middle_layer = 4 ∧
    s.bottom_layer = 9 →
    exposed_surface_area s = 33 := by
  sorry


end NUMINAMATH_CALUDE_sculpture_surface_area_l3722_372216


namespace NUMINAMATH_CALUDE_nail_count_proof_l3722_372234

/-- The number of nails Violet has -/
def violet_nails : ℕ := 27

/-- The number of nails Tickletoe has -/
def tickletoe_nails : ℕ := (violet_nails - 3) / 2

/-- The number of nails SillySocks has -/
def sillysocks_nails : ℕ := 3 * tickletoe_nails - 2

/-- The total number of nails -/
def total_nails : ℕ := violet_nails + tickletoe_nails + sillysocks_nails

theorem nail_count_proof :
  total_nails = 73 :=
sorry

end NUMINAMATH_CALUDE_nail_count_proof_l3722_372234


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l3722_372252

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ),
    (∀ x, f' x₀ ≤ f' x) ∧
    y₀ = f x₀ ∧
    (∀ x y, y = f x → 3*x - y - 2 = 0 ∨ 3*x - y - 2 > 0) ∧
    3*x₀ - y₀ - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l3722_372252


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3722_372274

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3722_372274


namespace NUMINAMATH_CALUDE_school_gender_difference_l3722_372256

theorem school_gender_difference (girls boys : ℕ) 
  (h1 : girls = 34) 
  (h2 : boys = 841) : 
  boys - girls = 807 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l3722_372256


namespace NUMINAMATH_CALUDE_penelope_savings_l3722_372206

theorem penelope_savings (daily_savings : ℕ) (total_savings : ℕ) (savings_period : ℕ) :
  daily_savings = 24 →
  total_savings = 8760 →
  savings_period * daily_savings = total_savings →
  savings_period = 365 := by
sorry

end NUMINAMATH_CALUDE_penelope_savings_l3722_372206


namespace NUMINAMATH_CALUDE_common_volume_theorem_l3722_372253

/-- Represents a triangular pyramid with a point O on the segment connecting
    the vertex with the intersection point of base medians -/
structure TriangularPyramid where
  volume : ℝ
  ratio : ℝ

/-- Calculates the volume of the common part of the original pyramid
    and its symmetric counterpart with respect to point O -/
noncomputable def commonVolume (pyramid : TriangularPyramid) : ℝ :=
  if pyramid.ratio = 1 then 2 * pyramid.volume / 9
  else if pyramid.ratio = 3 then pyramid.volume / 2
  else if pyramid.ratio = 2 then 110 * pyramid.volume / 243
  else if pyramid.ratio = 4 then 12 * pyramid.volume / 25
  else 0  -- undefined for other ratios

theorem common_volume_theorem (pyramid : TriangularPyramid) :
  (pyramid.ratio = 1 → commonVolume pyramid = 2 * pyramid.volume / 9) ∧
  (pyramid.ratio = 3 → commonVolume pyramid = pyramid.volume / 2) ∧
  (pyramid.ratio = 2 → commonVolume pyramid = 110 * pyramid.volume / 243) ∧
  (pyramid.ratio = 4 → commonVolume pyramid = 12 * pyramid.volume / 25) :=
by sorry

end NUMINAMATH_CALUDE_common_volume_theorem_l3722_372253


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3722_372266

theorem opposite_of_negative_one_third : 
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3722_372266


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_10201_l3722_372213

theorem largest_prime_factor_of_10201 : 
  (Nat.factors 10201).maximum? = some 37 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_10201_l3722_372213


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3722_372261

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l3722_372261


namespace NUMINAMATH_CALUDE_linear_regression_average_increase_l3722_372250

/- Define a linear regression model -/
def LinearRegression (x y : ℝ → ℝ) (a b : ℝ) :=
  ∀ t, y t = b * x t + a

/- Define the average increase in y when x increases by 1 unit -/
def AverageIncrease (x y : ℝ → ℝ) (b : ℝ) :=
  ∀ t, y (t + 1) - y t = b

/- Theorem: In a linear regression model, when x increases by 1 unit,
   y increases by b units on average -/
theorem linear_regression_average_increase
  (x y : ℝ → ℝ) (a b : ℝ)
  (h : LinearRegression x y a b) :
  AverageIncrease x y b :=
by sorry

end NUMINAMATH_CALUDE_linear_regression_average_increase_l3722_372250


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3722_372217

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 4 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3722_372217


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l3722_372281

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_function_theorem (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : f a b c a = a^3)
  (h2 : f a b c b = b^3) :
  c = 16 := by sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l3722_372281


namespace NUMINAMATH_CALUDE_smallest_root_property_l3722_372243

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x - 10 = 0

-- Define a as the smallest root
def a : ℝ := sorry

-- State the properties of a
axiom a_is_root : quadratic_equation a
axiom a_is_smallest : ∀ x, quadratic_equation x → a ≤ x

-- Theorem to prove
theorem smallest_root_property : a^4 - 909*a = 910 := by sorry

end NUMINAMATH_CALUDE_smallest_root_property_l3722_372243


namespace NUMINAMATH_CALUDE_count_words_with_vowels_l3722_372255

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 7

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words with at least one vowel -/
def words_with_vowels : ℕ := alphabet_size ^ word_length - (alphabet_size - vowel_count) ^ word_length

theorem count_words_with_vowels :
  words_with_vowels = 13682 :=
sorry

end NUMINAMATH_CALUDE_count_words_with_vowels_l3722_372255


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l3722_372285

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 3 4 5) ∧
  ¬(is_pythagorean_triple 3 4 7) ∧
  ¬(is_pythagorean_triple 0 1 1) ∧
  is_pythagorean_triple 9 12 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l3722_372285


namespace NUMINAMATH_CALUDE_initial_birds_count_l3722_372298

def birds_problem (initial_birds : ℕ) (landed_birds : ℕ) (total_birds : ℕ) : Prop :=
  initial_birds + landed_birds = total_birds

theorem initial_birds_count : ∃ (initial_birds : ℕ), 
  birds_problem initial_birds 8 20 ∧ initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l3722_372298


namespace NUMINAMATH_CALUDE_sum_maximum_l3722_372211

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  relation : 8 * a 5 = 13 * a 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating when the sum reaches its maximum -/
theorem sum_maximum (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_n seq n ≥ sum_n seq m ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_maximum_l3722_372211


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l3722_372283

/-- Given a rectangular box Q inscribed in a sphere of radius r,
    if the surface area of Q is 672 and the sum of the lengths of its 12 edges is 168,
    then r = √273 -/
theorem inscribed_box_radius (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  2 * (a * b + b * c + a * c) = 672 →
  4 * (a + b + c) = 168 →
  (2 * r) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 →
  r = Real.sqrt 273 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l3722_372283


namespace NUMINAMATH_CALUDE_astronaut_revolutions_l3722_372223

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of the three circles -/
structure CircleConfiguration where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  n : ℕ

/-- Defines the conditions of the problem -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  config.n > 2 ∧
  config.c₁.radius = config.n * config.c₃.radius ∧
  config.c₂.radius = 2 * config.c₃.radius

/-- Calculates the number of revolutions of c₃ relative to the ground -/
noncomputable def revolutions (config : CircleConfiguration) : ℝ :=
  config.n - 1

/-- The main theorem to be proved -/
theorem astronaut_revolutions 
  (config : CircleConfiguration) 
  (h : ValidConfiguration config) :
  revolutions config = config.n - 1 := by
  sorry

end NUMINAMATH_CALUDE_astronaut_revolutions_l3722_372223


namespace NUMINAMATH_CALUDE_earl_went_up_seven_floors_l3722_372227

/-- Represents the number of floors in the building -/
def total_floors : ℕ := 20

/-- Represents Earl's initial floor -/
def initial_floor : ℕ := 1

/-- Represents the number of floors Earl goes up initially -/
def first_up : ℕ := 5

/-- Represents the number of floors Earl goes down -/
def down : ℕ := 2

/-- Represents the number of floors Earl is away from the top after his final movement -/
def floors_from_top : ℕ := 9

/-- Calculates the number of floors Earl went up the second time -/
def second_up : ℕ := total_floors - floors_from_top - (initial_floor + first_up - down)

/-- Theorem stating that Earl went up 7 floors the second time -/
theorem earl_went_up_seven_floors : second_up = 7 := by sorry

end NUMINAMATH_CALUDE_earl_went_up_seven_floors_l3722_372227


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3722_372232

/-- Given a tank that requires different numbers of buckets to fill based on bucket capacity,
    this theorem proves the relationship between the original and reduced bucket capacities. -/
theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℕ) 
  (h1 : original_buckets = 10)
  (h2 : reduced_buckets = 25)
  : (original_buckets : ℚ) / reduced_buckets = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3722_372232


namespace NUMINAMATH_CALUDE_negation_of_exp_inequality_l3722_372201

theorem negation_of_exp_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) → 
  (¬p ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x < 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exp_inequality_l3722_372201


namespace NUMINAMATH_CALUDE_cannot_form_square_l3722_372251

/-- Represents the number of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total perimeter from the given stick counts -/
def totalPerimeter (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Checks if it's possible to form a square with the given stick counts -/
def canFormSquare (counts : StickCounts) : Prop :=
  ∃ (side : Nat), side > 0 ∧ 4 * side = totalPerimeter counts

/-- The given stick counts -/
def givenSticks : StickCounts :=
  { one_cm := 6
  , two_cm := 3
  , three_cm := 6
  , four_cm := 5
  }

/-- Theorem stating it's impossible to form a square with the given sticks -/
theorem cannot_form_square : ¬ canFormSquare givenSticks := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l3722_372251


namespace NUMINAMATH_CALUDE_acute_angle_through_point_l3722_372218

theorem acute_angle_through_point (α : Real) : 
  α > 0 ∧ α < Real.pi/2 →
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = Real.cos (40 * Real.pi/180) + 1 ∧ 
                            r * (Real.sin α) = Real.sin (40 * Real.pi/180)) →
  α = 20 * Real.pi/180 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_through_point_l3722_372218


namespace NUMINAMATH_CALUDE_x_percentage_of_y_pay_l3722_372207

/-- The percentage of Y's pay that X is paid, given the total pay and Y's pay -/
theorem x_percentage_of_y_pay (total_pay y_pay : ℝ) (h1 : total_pay = 700) (h2 : y_pay = 318.1818181818182) :
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_x_percentage_of_y_pay_l3722_372207


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3722_372271

theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 0.02 * r) = 600)
  (h2 : P * (1 + 0.07 * r) = 850) : 
  P = 500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3722_372271


namespace NUMINAMATH_CALUDE_plates_per_meal_l3722_372222

theorem plates_per_meal (guests : ℕ) (people : ℕ) (meals_per_day : ℕ) (days : ℕ) (total_plates : ℕ) :
  guests = 5 →
  people = guests + 1 →
  meals_per_day = 3 →
  days = 4 →
  total_plates = 144 →
  (total_plates / (people * meals_per_day * days) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_plates_per_meal_l3722_372222


namespace NUMINAMATH_CALUDE_total_selection_methods_l3722_372233

-- Define the number of candidate schools
def total_schools : ℕ := 8

-- Define the number of schools to be selected
def selected_schools : ℕ := 4

-- Define the number of schools for session A
def schools_in_session_A : ℕ := 2

-- Define the number of remaining sessions (B and C)
def remaining_sessions : ℕ := 2

-- Theorem to prove
theorem total_selection_methods :
  (total_schools.choose selected_schools) *
  (selected_schools.choose schools_in_session_A) *
  (remaining_sessions!) = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_methods_l3722_372233


namespace NUMINAMATH_CALUDE_expected_winnings_is_negative_half_dollar_l3722_372248

/-- Represents the sections of the spinner --/
inductive Section
  | Red
  | Blue
  | Green
  | Yellow

/-- Returns the probability of landing on a given section --/
def probability (s : Section) : ℚ :=
  match s with
  | Section.Red => 3/8
  | Section.Blue => 1/4
  | Section.Green => 1/4
  | Section.Yellow => 1/8

/-- Returns the winnings (in dollars) for a given section --/
def winnings (s : Section) : ℤ :=
  match s with
  | Section.Red => 2
  | Section.Blue => 4
  | Section.Green => -3
  | Section.Yellow => -6

/-- Calculates the expected winnings from spinning the spinner --/
def expectedWinnings : ℚ :=
  (probability Section.Red * winnings Section.Red) +
  (probability Section.Blue * winnings Section.Blue) +
  (probability Section.Green * winnings Section.Green) +
  (probability Section.Yellow * winnings Section.Yellow)

/-- Theorem stating that the expected winnings is -$0.50 --/
theorem expected_winnings_is_negative_half_dollar :
  expectedWinnings = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_negative_half_dollar_l3722_372248


namespace NUMINAMATH_CALUDE_craig_commission_l3722_372229

/-- Calculates the total commission for Craig's appliance sales. -/
def total_commission (
  refrigerator_base : ℝ)
  (refrigerator_rate : ℝ)
  (washing_machine_base : ℝ)
  (washing_machine_rate : ℝ)
  (oven_base : ℝ)
  (oven_rate : ℝ)
  (refrigerator_count : ℕ)
  (refrigerator_total_price : ℝ)
  (washing_machine_count : ℕ)
  (washing_machine_total_price : ℝ)
  (oven_count : ℕ)
  (oven_total_price : ℝ) : ℝ :=
  (refrigerator_count * (refrigerator_base + refrigerator_rate * refrigerator_total_price)) +
  (washing_machine_count * (washing_machine_base + washing_machine_rate * washing_machine_total_price)) +
  (oven_count * (oven_base + oven_rate * oven_total_price))

/-- Craig's total commission for the week is $5620.20. -/
theorem craig_commission :
  total_commission 75 0.08 50 0.10 60 0.12 3 5280 4 2140 5 4620 = 5620.20 := by
  sorry

end NUMINAMATH_CALUDE_craig_commission_l3722_372229


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3722_372269

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x * (x - 5) < 0) ∧
  (∃ x : ℝ, x * (x - 5) < 0 ∧ ¬(2 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3722_372269


namespace NUMINAMATH_CALUDE_tank_overflow_time_l3722_372236

/-- Represents the time it takes for a pipe to fill the tank -/
structure PipeRate where
  fill_time : ℝ
  fill_time_pos : fill_time > 0

/-- Represents the state of the tank filling process -/
structure TankFilling where
  overflow_time : ℝ
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_b_close_time : ℝ

/-- The main theorem stating when the tank will overflow -/
theorem tank_overflow_time (tf : TankFilling) 
  (h1 : tf.pipe_a.fill_time = 2)
  (h2 : tf.pipe_b.fill_time = 1)
  (h3 : tf.pipe_b_close_time = tf.overflow_time - 0.5)
  (h4 : tf.overflow_time > 0) :
  tf.overflow_time = 1 := by
  sorry

#check tank_overflow_time

end NUMINAMATH_CALUDE_tank_overflow_time_l3722_372236


namespace NUMINAMATH_CALUDE_marie_message_clearing_l3722_372202

/-- Calculate the number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  if read_per_day > new_per_day then
    (initial_messages + (read_per_day - new_per_day - 1)) / (read_per_day - new_per_day)
  else
    0

theorem marie_message_clearing :
  days_to_clear_messages 98 20 6 = 7 := by
sorry

end NUMINAMATH_CALUDE_marie_message_clearing_l3722_372202


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3722_372247

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3722_372247


namespace NUMINAMATH_CALUDE_probability_of_winning_all_games_l3722_372244

def number_of_games : ℕ := 6
def probability_of_winning_single_game : ℚ := 3/5

theorem probability_of_winning_all_games :
  (probability_of_winning_single_game ^ number_of_games : ℚ) = 729/15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_all_games_l3722_372244


namespace NUMINAMATH_CALUDE_unique_solution_natural_equation_l3722_372294

theorem unique_solution_natural_equation :
  ∀ (a b x y : ℕ),
    x^(a + b) + y = x^a * y^b →
    (x = 2 ∧ y = 4) ∧ (∀ (x' y' : ℕ), x'^(a + b) + y' = x'^a * y'^b → x' = 2 ∧ y' = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_natural_equation_l3722_372294


namespace NUMINAMATH_CALUDE_min_total_faces_l3722_372280

/-- Represents a fair die with a given number of faces. -/
structure Die where
  faces : ℕ
  faces_gt_6 : faces > 6

/-- Calculates the number of ways to roll a given sum with two dice. -/
def waysToRoll (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- The probability of rolling a given sum with two dice. -/
def probOfSum (d1 d2 : Die) (sum : ℕ) : ℚ :=
  sorry

theorem min_total_faces (d1 d2 : Die) :
  (probOfSum d1 d2 8 = (1 : ℚ) / 2 * probOfSum d1 d2 11) →
  (probOfSum d1 d2 15 = (1 : ℚ) / 30) →
  d1.faces + d2.faces ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_total_faces_l3722_372280


namespace NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l3722_372258

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem angle_not_sharing_terminal_side :
  ¬(same_terminal_side 680 (-750)) ∧
  (same_terminal_side 330 (-750)) ∧
  (same_terminal_side (-30) (-750)) ∧
  (same_terminal_side (-1110) (-750)) :=
sorry

end NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l3722_372258


namespace NUMINAMATH_CALUDE_three_quarters_of_fifteen_fifths_minus_half_l3722_372288

theorem three_quarters_of_fifteen_fifths_minus_half (x : ℚ) : x = (3 / 4) * (15 / 5) - (1 / 2) → x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_of_fifteen_fifths_minus_half_l3722_372288


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3722_372284

theorem arithmetic_calculations :
  ((-24) - (-15) + (-1) + (-15) = -25) ∧
  ((-27) / (3/2) * (2/3) = -12) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3722_372284


namespace NUMINAMATH_CALUDE_coefficient_x4_l3722_372230

/-- The coefficient of x^4 in the simplified form of 5(x^4 - 3x^2) + 3(2x^3 - x^4 + 4x^6) - (6x^2 - 2x^4) is 4 -/
theorem coefficient_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 3*x^2) + 3*(2*x^3 - x^4 + 4*x^6) - (6*x^2 - 2*x^4)
  ∃ (a b c d e : ℝ), expr = 4*x^4 + a*x^6 + b*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x4_l3722_372230


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l3722_372267

theorem max_sum_squared_distances (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  (∃ (max_val : ℝ), max_val = 15 + 24 * (1.5 / Real.sqrt (1.5^2 + 1)) - 16 * (1 / Real.sqrt (1.5^2 + 1)) ∧
   ∀ (w : ℂ), Complex.abs (w - (3 - 3*I)) = 4 →
     Complex.abs (w - (2 + I))^2 + Complex.abs (w - (6 - 2*I))^2 ≤ max_val) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l3722_372267


namespace NUMINAMATH_CALUDE_logan_hair_length_l3722_372228

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Logan's hair length. -/
theorem logan_hair_length (kate_length emily_length logan_length : ℝ) 
  (h1 : kate_length = 7)
  (h2 : kate_length = emily_length / 2)
  (h3 : emily_length = logan_length + 6) :
  logan_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_logan_hair_length_l3722_372228


namespace NUMINAMATH_CALUDE_min_value_expression_l3722_372219

/-- The minimum value of a specific expression given certain constraints -/
theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) 
  (h_prod : x * y * z * w = 3) :
  x^2 + 4*x*y + 9*y^2 + 6*y*z + 8*z^2 + 3*x*w + 4*w^2 ≥ 81.25 ∧ 
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ w₀ > 0 ∧ 
    x₀ * y₀ * z₀ * w₀ = 3 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 6*y₀*z₀ + 8*z₀^2 + 3*x₀*w₀ + 4*w₀^2 = 81.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3722_372219


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3722_372221

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
  let B : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  (A ∪ B) ⊆ (A ∩ B) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l3722_372221


namespace NUMINAMATH_CALUDE_festival_attendance_l3722_372299

theorem festival_attendance (total : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) 
  (h_total : total = 2700)
  (h_day2 : day2 = day1 / 2)
  (h_day3 : day3 = 3 * day1)
  (h_sum : day1 + day2 + day3 = total) :
  day2 = 300 := by
sorry

end NUMINAMATH_CALUDE_festival_attendance_l3722_372299


namespace NUMINAMATH_CALUDE_no_common_root_l3722_372235

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ∀ x : ℝ, (x^2 + b*x + c = 0) → (x^2 + a*x + d = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_l3722_372235


namespace NUMINAMATH_CALUDE_two_interviewers_passing_l3722_372291

def number_of_interviewers : ℕ := 5
def interviewers_to_choose : ℕ := 2

theorem two_interviewers_passing :
  Nat.choose number_of_interviewers interviewers_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_interviewers_passing_l3722_372291


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l3722_372287

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ x₁^2 - 5*x₁ - 6 = 0 ∧ x₂^2 - 5*x₂ - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l3722_372287


namespace NUMINAMATH_CALUDE_sophomore_count_l3722_372262

theorem sophomore_count (total : ℕ) (soph_percent : ℚ) (junior_percent : ℚ) :
  total = 36 →
  soph_percent = 1/5 →
  junior_percent = 3/20 →
  ∃ (soph junior : ℕ),
    soph + junior = total ∧
    soph_percent * soph = junior_percent * junior ∧
    soph = 16 :=
by sorry

end NUMINAMATH_CALUDE_sophomore_count_l3722_372262


namespace NUMINAMATH_CALUDE_vector_problem_l3722_372249

/-- The angle between two 2D vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Checks if two 2D vectors are collinear -/
def collinear (v w : ℝ × ℝ) : Prop := sorry

/-- Checks if two 2D vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop := sorry

theorem vector_problem (a b c : ℝ × ℝ) 
  (ha : a = (1, 2))
  (hb : b = (-2, 6))
  (hc : c = (-1, 3)) : 
  angle a b = π/4 ∧ 
  collinear b c ∧ 
  perpendicular a (a - c) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3722_372249


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l3722_372279

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def candies_left : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

/-- The number of candies Susan ate -/
def candies_eaten : ℕ := total_candies - candies_left

theorem susan_ate_six_candies : candies_eaten = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l3722_372279


namespace NUMINAMATH_CALUDE_arg_z1_div_z2_l3722_372289

theorem arg_z1_div_z2 (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 1) (h3 : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arg_z1_div_z2_l3722_372289


namespace NUMINAMATH_CALUDE_x_142_equals_1995_unique_l3722_372246

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := 
  if p x = 2 then 1 else sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n * p (x n) / q (x n)

theorem x_142_equals_1995_unique : 
  (x 142 = 1995) ∧ (∀ n : ℕ, n ≠ 142 → x n ≠ 1995) := by sorry

end NUMINAMATH_CALUDE_x_142_equals_1995_unique_l3722_372246


namespace NUMINAMATH_CALUDE_train_distance_l3722_372214

theorem train_distance (v_ab v_ba : ℝ) (t_diff : ℝ) (h1 : v_ab = 160)
    (h2 : v_ba = 120) (h3 : t_diff = 1) : ∃ D : ℝ,
  D / v_ba = D / v_ab + t_diff ∧ D = 480 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3722_372214


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_necessity_not_sufficient_l3722_372282

/-- Defines an ellipse in terms of its equation coefficients -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is necessary but not sufficient for mx^2 + ny^2 = 1 to be an ellipse -/
theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

/-- Proving necessity: if mx^2 + ny^2 = 1 is an ellipse, then mn > 0 -/
theorem necessity (m n : ℝ) (h : is_ellipse m n) : m * n > 0 :=
sorry

/-- Proving not sufficient: there exist m and n where mn > 0 but mx^2 + ny^2 = 1 is not an ellipse -/
theorem not_sufficient : ∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_necessity_not_sufficient_l3722_372282


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3722_372237

/-- A sequence of natural numbers -/
def IncreasingSequence := ℕ → ℕ

/-- Property that the sequence is strictly increasing -/
def IsStrictlyIncreasing (a : IncreasingSequence) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a (n + 1) > a n

/-- Property that every natural number is the sum of two sequence terms -/
def HasAllSums (a : IncreasingSequence) : Prop :=
  ∀ k : ℕ, ∃ i j : ℕ, k = a i + a j

/-- Property that each term is greater than n²/16 -/
def SatisfiesLowerBound (a : IncreasingSequence) : Prop :=
  ∀ n : ℕ, n > 0 → a n > (n^2 : ℚ) / 16

/-- The main theorem stating the existence of a sequence satisfying all conditions -/
theorem exists_special_sequence :
  ∃ a : IncreasingSequence, 
    IsStrictlyIncreasing a ∧ 
    HasAllSums a ∧ 
    SatisfiesLowerBound a := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l3722_372237


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l3722_372245

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define the complement of A with respect to U
def complement_A : Finset Nat := {2}

-- Define set A based on its complement
def A : Finset Nat := U \ complement_A

-- Theorem statement
theorem number_of_proper_subsets_of_A : 
  Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l3722_372245


namespace NUMINAMATH_CALUDE_negative_integers_abs_leq_four_l3722_372239

theorem negative_integers_abs_leq_four :
  {x : ℤ | x < 0 ∧ |x| ≤ 4} = {-1, -2, -3, -4} := by sorry

end NUMINAMATH_CALUDE_negative_integers_abs_leq_four_l3722_372239


namespace NUMINAMATH_CALUDE_set_c_is_proportional_l3722_372260

/-- A set of four real numbers is proportional if the product of its extremes equals the product of its means -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set (2, 3, 4, 6) is proportional -/
theorem set_c_is_proportional : isProportional 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_set_c_is_proportional_l3722_372260
