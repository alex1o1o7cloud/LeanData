import Mathlib

namespace num_divisors_23232_l2797_279798

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- 23232 as a positive integer -/
def n : ℕ+ := 23232

/-- Theorem stating that the number of positive divisors of 23232 is 42 -/
theorem num_divisors_23232 : num_divisors n = 42 := by sorry

end num_divisors_23232_l2797_279798


namespace arithmetic_calculation_l2797_279771

theorem arithmetic_calculation : (30 / (10 + 2 - 5) + 4) * 7 = 58 := by
  sorry

end arithmetic_calculation_l2797_279771


namespace area_difference_l2797_279780

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end area_difference_l2797_279780


namespace new_person_weight_l2797_279782

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 98.6 kg -/
theorem new_person_weight :
  weight_of_new_person 8 4.2 65 = 98.6 := by
  sorry

#eval weight_of_new_person 8 4.2 65

end new_person_weight_l2797_279782


namespace room_occupancy_l2797_279703

theorem room_occupancy (x y : ℕ) : 
  x + y = 76 → 
  x - 30 = y - 40 → 
  (x = 33 ∧ y = 43) ∨ (x = 43 ∧ y = 33) :=
by sorry

end room_occupancy_l2797_279703


namespace equation_represents_empty_set_l2797_279765

theorem equation_represents_empty_set : 
  ∀ (x y : ℝ), 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y ≠ 0 := by
  sorry

end equation_represents_empty_set_l2797_279765


namespace jane_rejection_proof_l2797_279763

/-- Represents the percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.9

/-- Represents the percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.5

/-- Represents the fraction of total products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.625

/-- Represents the total percentage of rejected products -/
def total_rejection_rate : ℝ := 0.75

/-- Theorem stating that given the conditions, Jane's rejection rate is 0.9% -/
theorem jane_rejection_proof :
  john_rejection_rate * (1 - jane_inspection_fraction) +
  jane_rejection_rate * jane_inspection_fraction / 100 =
  total_rejection_rate / 100 := by
  sorry

#check jane_rejection_proof

end jane_rejection_proof_l2797_279763


namespace man_swimming_speed_l2797_279796

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 1 km/h. -/
def swimming_speed : ℝ := 2

/-- The speed of the stream in km/h. -/
def stream_speed : ℝ := 1

/-- The time ratio of swimming upstream to downstream. -/
def upstream_downstream_ratio : ℝ := 2

theorem man_swimming_speed :
  swimming_speed = 2 ∧
  stream_speed = 1 ∧
  upstream_downstream_ratio = 2 →
  swimming_speed + stream_speed = upstream_downstream_ratio * (swimming_speed - stream_speed) :=
by sorry

end man_swimming_speed_l2797_279796


namespace complement_of_A_l2797_279728

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 0 < x ∧ x < 1/3}

-- Define the complement of A with respect to U
def complementU (A : Set ℝ) : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A : complementU A = {x | x = 0 ∨ 1/3 ≤ x ∧ x ≤ 1} := by
  sorry

end complement_of_A_l2797_279728


namespace max_b_value_l2797_279724

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c ≤ b →
  b < a →
  b ≤ 10 :=
by sorry

end max_b_value_l2797_279724


namespace power_product_equals_sum_of_exponents_l2797_279790

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l2797_279790


namespace least_subtrahend_l2797_279705

def is_valid (x : ℕ) : Prop :=
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3

theorem least_subtrahend :
  ∃ (x : ℕ), is_valid x ∧ ∀ (y : ℕ), y < x → ¬is_valid y :=
by sorry

end least_subtrahend_l2797_279705


namespace honey_production_l2797_279741

/-- The amount of honey (in grams) produced by a single bee in 60 days -/
def single_bee_honey : ℕ := 1

/-- The number of bees in the group -/
def num_bees : ℕ := 60

/-- The amount of honey (in grams) produced by the group of bees in 60 days -/
def group_honey : ℕ := num_bees * single_bee_honey

theorem honey_production :
  group_honey = 60 := by sorry

end honey_production_l2797_279741


namespace total_tiles_is_44_l2797_279754

-- Define the room dimensions
def room_length : ℕ := 20
def room_width : ℕ := 15

-- Define tile sizes
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Function to calculate the number of border tiles
def border_tiles : ℕ :=
  2 * (room_length / border_tile_size + room_width / border_tile_size) - 4

-- Function to calculate the inner area
def inner_area : ℕ :=
  (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)

-- Function to calculate the number of inner tiles
def inner_tiles : ℕ :=
  (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

-- Theorem stating the total number of tiles
theorem total_tiles_is_44 :
  border_tiles + inner_tiles = 44 := by
  sorry

end total_tiles_is_44_l2797_279754


namespace motorcycle_cyclist_meeting_times_l2797_279743

theorem motorcycle_cyclist_meeting_times 
  (angle : Real) 
  (cyclist_speed : Real) 
  (motorcyclist_speed : Real) 
  (t : Real) : 
  angle = π / 3 →
  cyclist_speed = 36 →
  motorcyclist_speed = 72 →
  (motorcyclist_speed^2 * t^2 + cyclist_speed^2 * (t - 1)^2 - 
   2 * motorcyclist_speed * cyclist_speed * |t| * |t - 1| * (1/2) = 252^2) →
  (t = 4 ∨ t = -4) :=
by sorry

end motorcycle_cyclist_meeting_times_l2797_279743


namespace domino_placement_l2797_279764

/-- The maximum number of 1 × k dominos that can be placed on an n × n chessboard. -/
def max_dominos (n k : ℕ) : ℕ :=
  if n = k ∨ n = 2*k - 1 then n
  else if k < n ∧ n < 2*k - 1 then 2*n - 2*k + 2
  else 0

theorem domino_placement (n k : ℕ) (h1 : k ≤ n) (h2 : n < 2*k) :
  max_dominos n k = if n = k ∨ n = 2*k - 1 then n else 2*n - 2*k + 2 :=
by sorry

end domino_placement_l2797_279764


namespace division_ways_correct_l2797_279713

/-- The number of ways to divide 6 distinct objects into three groups,
    where one group has 4 objects and the other two groups have 1 object each. -/
def divisionWays : ℕ := 15

/-- The total number of objects to be divided. -/
def totalObjects : ℕ := 6

/-- The number of objects in the largest group. -/
def largestGroupSize : ℕ := 4

/-- The number of groups. -/
def numberOfGroups : ℕ := 3

/-- Theorem stating that the number of ways to divide the objects is correct. -/
theorem division_ways_correct :
  divisionWays = Nat.choose totalObjects largestGroupSize :=
sorry

end division_ways_correct_l2797_279713


namespace complex_equation_sum_l2797_279772

theorem complex_equation_sum (x y : ℝ) :
  (x - Complex.I) * Complex.I = y + 2 * Complex.I →
  x + y = 3 := by
  sorry

end complex_equation_sum_l2797_279772


namespace chloe_trivia_game_score_l2797_279777

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_game_score (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : final_score = 86) :
  (first_round + second_round) - final_score = 4 := by
  sorry

end chloe_trivia_game_score_l2797_279777


namespace multiple_properties_l2797_279760

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) :=
by sorry

end multiple_properties_l2797_279760


namespace distance_AE_BF_is_19_2_l2797_279769

/-- A rectangular parallelepiped with given dimensions and midpoints -/
structure Parallelepiped where
  -- Edge lengths
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  -- Ensure it's a rectangular parallelepiped
  is_rectangular : True
  -- Ensure E is midpoint of A₁B₁
  e_is_midpoint_a1b1 : True
  -- Ensure F is midpoint of B₁C₁
  f_is_midpoint_b1c1 : True

/-- The distance between lines AE and BF in the parallelepiped -/
def distance_AE_BF (p : Parallelepiped) : ℝ := sorry

/-- Theorem: The distance between AE and BF is 19.2 -/
theorem distance_AE_BF_is_19_2 (p : Parallelepiped) 
  (h1 : p.ab = 30) (h2 : p.ad = 32) (h3 : p.aa1 = 20) : 
  distance_AE_BF p = 19.2 := by sorry

end distance_AE_BF_is_19_2_l2797_279769


namespace extremal_point_implies_k_range_l2797_279727

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) + 2*k*(Real.log x) - k*x

theorem extremal_point_implies_k_range :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (x ≠ 2 → (deriv (f k)) x ≠ 0)) →
  k ∈ Set.Iic ((Real.exp 2) / 4) :=
sorry

end extremal_point_implies_k_range_l2797_279727


namespace cannot_obtain_703_from_604_l2797_279719

/-- Represents the computer operations -/
inductive Operation
  | square : Operation
  | split : Operation

/-- Applies the given operation to a natural number -/
def apply_operation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.split => 
      if n < 1000 then n
      else (n % 1000) + (n / 1000)

/-- Checks if it's possible to transform start into target using the given operations -/
def can_transform (start target : ℕ) : Prop :=
  ∃ (seq : List Operation), 
    (seq.foldl (λ acc op => apply_operation op acc) start) = target

/-- The main theorem stating that 703 cannot be obtained from 604 using the given operations -/
theorem cannot_obtain_703_from_604 : ¬ can_transform 604 703 := by
  sorry


end cannot_obtain_703_from_604_l2797_279719


namespace intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l2797_279799

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem k_range_when_intersection_nonempty :
  ∀ k : ℝ, (A ∩ B k).Nonempty → k ≥ -1 := by sorry

end intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l2797_279799


namespace paving_stone_width_l2797_279752

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 1 meter. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16)
  (h3 : stone_length = 2)
  (h4 : stone_count = 240)
  : ∃ (stone_width : ℝ), 
    stone_width = 1 ∧ 
    courtyard_length * courtyard_width = ↑stone_count * stone_length * stone_width :=
by sorry

end paving_stone_width_l2797_279752


namespace zero_point_of_f_l2797_279789

def f (x : ℝ) : ℝ := x + 1

theorem zero_point_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by
  sorry

end zero_point_of_f_l2797_279789


namespace bus_wheel_radius_l2797_279797

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 70.06369426751593) : 
  ∃ (r : ℝ), abs (r - 2500.57) < 0.01 := by
  sorry

end bus_wheel_radius_l2797_279797


namespace vectors_are_coplanar_l2797_279735

def a : ℝ × ℝ × ℝ := (1, -2, 6)
def b : ℝ × ℝ × ℝ := (1, 0, 1)
def c : ℝ × ℝ × ℝ := (2, -6, 17)

def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (u₁ * (v₂ * w₃ - v₃ * w₂) - u₂ * (v₁ * w₃ - v₃ * w₁) + u₃ * (v₁ * w₂ - v₂ * w₁)) = 0

theorem vectors_are_coplanar : coplanar a b c := by
  sorry

end vectors_are_coplanar_l2797_279735


namespace line_equation_60_degrees_l2797_279740

/-- The equation of a line with a slope of 60° and a y-intercept of -1 -/
theorem line_equation_60_degrees (x y : ℝ) :
  let slope : ℝ := Real.tan (60 * π / 180)
  let y_intercept : ℝ := -1
  slope * x - y - y_intercept = 0 ↔ Real.sqrt 3 * x - y - 1 = 0 := by
  sorry

end line_equation_60_degrees_l2797_279740


namespace equation_value_l2797_279773

theorem equation_value (x y : ℝ) (h : x - 3*y = 4) : 
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end equation_value_l2797_279773


namespace vector_decomposition_l2797_279716

def x : Fin 3 → ℝ := ![6, -1, 7]
def p : Fin 3 → ℝ := ![1, -2, 0]
def q : Fin 3 → ℝ := ![-1, 1, 3]
def r : Fin 3 → ℝ := ![1, 0, 4]

theorem vector_decomposition :
  x = λ i => -p i - 3 * q i + 4 * r i :=
by sorry

end vector_decomposition_l2797_279716


namespace bound_difference_for_elements_in_A_l2797_279770

/-- The function f(x) = |x+2| + |x-2| -/
def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

/-- The set A of all x such that f(x) ≤ 6 -/
def A : Set ℝ := {x | f x ≤ 6}

/-- Theorem stating that if m and n are in A, then |1/3 * m - 1/2 * n| ≤ 5/2 -/
theorem bound_difference_for_elements_in_A (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end bound_difference_for_elements_in_A_l2797_279770


namespace general_term_correct_l2797_279751

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))
  S_3 : S 3 = 13/9
  S_6 : S 6 = 364/9

/-- The general term of the geometric sequence -/
def general_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  (1/6) * 3^(n-1)

/-- Theorem stating that the general term is correct -/
theorem general_term_correct (seq : GeometricSequence) :
  ∀ n : ℕ, seq.a n = general_term seq n :=
sorry

end general_term_correct_l2797_279751


namespace marys_potatoes_l2797_279759

def potatoes_problem (initial_potatoes : ℕ) (eaten_potatoes : ℕ) : Prop :=
  initial_potatoes - eaten_potatoes = 5

theorem marys_potatoes : potatoes_problem 8 3 :=
sorry

end marys_potatoes_l2797_279759


namespace function_f_form_l2797_279714

/-- A function from positive integers to non-negative integers satisfying the given property -/
def FunctionF (f : ℕ+ → ℕ) : Prop :=
  f ≠ 0 ∧ ∀ a b : ℕ+, 2 * f (a * b) = (↑b + 1) * f a + (↑a + 1) * f b

/-- The main theorem stating the existence of c such that f(n) = c(n-1) -/
theorem function_f_form (f : ℕ+ → ℕ) (hf : FunctionF f) :
  ∃ c : ℕ, ∀ n : ℕ+, f n = c * (↑n - 1) :=
sorry

end function_f_form_l2797_279714


namespace essay_section_length_l2797_279787

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_length : ℕ) 
  (body_sections : ℕ) 
  (total_length : ℕ) :
  intro_length = 450 →
  conclusion_length = 3 * intro_length →
  body_sections = 4 →
  total_length = 5000 →
  (total_length - (intro_length + conclusion_length)) / body_sections = 800 :=
by
  sorry

end essay_section_length_l2797_279787


namespace smallest_divisible_by_4_13_7_l2797_279701

theorem smallest_divisible_by_4_13_7 : ∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 364 := by
  sorry

#check smallest_divisible_by_4_13_7

end smallest_divisible_by_4_13_7_l2797_279701


namespace smallest_integer_l2797_279788

theorem smallest_integer (a b : ℕ) (ha : a = 60) (hb : b > 0) 
  (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  (∀ c : ℕ, c > 0 ∧ c < b → ¬(Nat.lcm a c / Nat.gcd a c = 60)) → b = 16 := by
  sorry

end smallest_integer_l2797_279788


namespace middle_number_not_unique_l2797_279712

/-- Represents a configuration of three cards with positive integers. -/
structure CardConfiguration where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_15 : left + middle + right = 15
  increasing : left < middle ∧ middle < right

/-- Predicate to check if Alan can determine the other two numbers. -/
def alan_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.left = config.left ∧ other_config ≠ config

/-- Predicate to check if Carlos can determine the other two numbers. -/
def carlos_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.right = config.right ∧ other_config ≠ config

/-- Predicate to check if Brenda can determine the other two numbers. -/
def brenda_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.middle = config.middle ∧ other_config ≠ config

/-- The main theorem stating that the middle number cannot be uniquely determined. -/
theorem middle_number_not_unique : ∃ (config1 config2 : CardConfiguration),
  config1.middle ≠ config2.middle ∧
  alan_cant_determine config1 ∧
  alan_cant_determine config2 ∧
  carlos_cant_determine config1 ∧
  carlos_cant_determine config2 ∧
  brenda_cant_determine config1 ∧
  brenda_cant_determine config2 :=
sorry

end middle_number_not_unique_l2797_279712


namespace brenda_money_brenda_money_proof_l2797_279750

/-- Proof that Brenda has 8 dollars given the conditions about Emma, Daya, Jeff, and Brenda's money. -/
theorem brenda_money : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun emma daya jeff brenda =>
    emma = 8 ∧
    daya = emma * 1.25 ∧
    jeff = daya * (2/5) ∧
    brenda = jeff + 4 →
    brenda = 8

-- Proof
theorem brenda_money_proof : brenda_money 8 10 4 8 := by
  sorry

end brenda_money_brenda_money_proof_l2797_279750


namespace sin_special_angle_l2797_279721

/-- Given a function f(x) = sin(x/2 + π/4), prove that f(π/2) = 1 -/
theorem sin_special_angle (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (x / 2 + π / 4)) :
  f (π / 2) = 1 := by
  sorry

end sin_special_angle_l2797_279721


namespace cos_pi_sixth_plus_alpha_l2797_279706

theorem cos_pi_sixth_plus_alpha (α : Real) 
  (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (π/6 + α) = -1/3 := by
  sorry

end cos_pi_sixth_plus_alpha_l2797_279706


namespace project_completion_time_l2797_279748

/-- The number of days it takes B to complete the project alone -/
def B_days : ℝ := 30

/-- The number of days A and B work together -/
def AB_work_days : ℝ := 10

/-- The number of days B works alone after A quits -/
def B_alone_days : ℝ := 5

/-- The number of days it takes A to complete the project alone -/
def A_days : ℝ := 20

theorem project_completion_time :
  (AB_work_days * (1 / A_days + 1 / B_days) + B_alone_days * (1 / B_days)) = 1 :=
sorry

end project_completion_time_l2797_279748


namespace inequality_system_solution_set_l2797_279722

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x > -6 - 2*x ∧ x ≤ (3 + x) / 4}
  S = {x | -2 < x ∧ x ≤ 1} := by
sorry

end inequality_system_solution_set_l2797_279722


namespace inverse_matrices_sum_l2797_279707

/-- Two 3x3 matrices that are inverses of each other -/
def matrix1 (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, b; 2, 2, 3; c, 5, d]
def matrix2 (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-5, e, -11; f, -13, g; 2, h, 4]

/-- The theorem stating that the sum of all variables is 45 -/
theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (matrix1 a b c d) * (matrix2 e f g h) = 1 →
  a + b + c + d + e + f + g + h = 45 := by
  sorry

end inverse_matrices_sum_l2797_279707


namespace jorge_total_goals_l2797_279779

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end jorge_total_goals_l2797_279779


namespace parallelogram_side_sum_l2797_279739

theorem parallelogram_side_sum (x y : ℝ) : 
  (12 * y - 2 = 10) → (5 * x + 15 = 20) → x + y = 2 := by
  sorry

end parallelogram_side_sum_l2797_279739


namespace store_inventory_sale_l2797_279778

theorem store_inventory_sale (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (debt : ℝ) (leftover : ℝ) :
  total_items = 2000 →
  original_price = 50 →
  discount_percent = 80 →
  debt = 15000 →
  leftover = 3000 →
  (((debt + leftover) / (original_price * (1 - discount_percent / 100))) / total_items) * 100 = 90 := by
  sorry


end store_inventory_sale_l2797_279778


namespace fraction_product_simplification_l2797_279725

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l2797_279725


namespace income_expenditure_ratio_l2797_279767

def income : ℕ := 10000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 10 / 7 := by sorry

end income_expenditure_ratio_l2797_279767


namespace jana_kelly_height_difference_l2797_279702

/-- Proves that Jana is 5 inches taller than Kelly given the heights of Jess and Jana, and the height difference between Jess and Kelly. -/
theorem jana_kelly_height_difference :
  ∀ (jess_height jana_height kelly_height : ℕ),
    jess_height = 72 →
    jana_height = 74 →
    kelly_height = jess_height - 3 →
    jana_height - kelly_height = 5 := by
  sorry

end jana_kelly_height_difference_l2797_279702


namespace solution_implies_sum_equals_four_l2797_279736

-- Define the operation ⊗
def otimes (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem solution_implies_sum_equals_four 
  (h : ∀ x : ℝ, (otimes (x - a) (x - b) > 0) ↔ (2 < x ∧ x < 3)) :
  a + b = 4 := by
  sorry

end solution_implies_sum_equals_four_l2797_279736


namespace production_theorem_l2797_279795

-- Define production lines
structure ProductionLine where
  process_rate : ℝ → ℝ
  inv_process_rate : ℝ → ℝ

-- Define the company
structure Company where
  line_A : ProductionLine
  line_B : ProductionLine

-- Define the problem
def production_problem (c : Company) : Prop :=
  -- Line A processes a tons in (4a+1) hours
  (c.line_A.process_rate = fun a => 4 * a + 1) ∧
  (c.line_A.inv_process_rate = fun t => (t - 1) / 4) ∧
  -- Line B processes b tons in (2b+3) hours
  (c.line_B.process_rate = fun b => 2 * b + 3) ∧
  (c.line_B.inv_process_rate = fun t => (t - 3) / 2) ∧
  -- Day 1: 5 tons allocated with equal processing time
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ c.line_A.process_rate x = c.line_B.process_rate (5 - x) ∧
  -- Day 2: 5 tons allocated based on day 1 results, plus m to A and n to B
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
    c.line_A.process_rate (x + m) = c.line_B.process_rate (5 - x + n) ∧
    c.line_A.process_rate (x + m) ≤ 24 ∧ c.line_B.process_rate (5 - x + n) ≤ 24

-- Theorem to prove
theorem production_theorem (c : Company) :
  production_problem c →
  (∃ (x : ℝ), x = 2 ∧ 5 - x = 3) ∧
  (∃ (m n : ℝ), m / n = 1 / 2) :=
by sorry

end production_theorem_l2797_279795


namespace constant_phi_forms_cone_l2797_279792

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  ∃ (cone : Set SphericalPoint), ConstantPhiSet d = cone :=
sorry

end constant_phi_forms_cone_l2797_279792


namespace combined_cost_theorem_l2797_279729

/-- Calculate the total cost of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCost (faceValue : ℚ) (discountPremiumRate : ℚ) (brokerageRate : ℚ) : ℚ :=
  let adjustedValue := faceValue * (1 + discountPremiumRate)
  adjustedValue * (1 + brokerageRate)

/-- The combined cost of stocks A, B, and C -/
def combinedCost : ℚ :=
  stockCost 100 (-0.02) (1/500) +  -- Stock A
  stockCost 150 0.015 (1/600) +    -- Stock B
  stockCost 200 (-0.03) (1/200)    -- Stock C

theorem combined_cost_theorem :
  combinedCost = 445669750/1000000 := by sorry

end combined_cost_theorem_l2797_279729


namespace sqrt_product_simplification_l2797_279720

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (10 * p) * Real.sqrt (5 * p^2) * Real.sqrt (6 * p^4) = 10 * p^3 * Real.sqrt (3 * p) := by
  sorry

end sqrt_product_simplification_l2797_279720


namespace oscars_voting_problem_l2797_279774

/-- Represents a film critic's vote --/
structure Vote where
  actor : Nat
  actress : Nat

/-- The problem statement --/
theorem oscars_voting_problem 
  (critics : Finset Vote) 
  (h_count : critics.card = 3366)
  (h_unique : ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 → ∃ v : Vote, (critics.filter (λ x => x.actor = v.actor ∨ x.actress = v.actress)).card = n) :
  ∃ v1 v2 : Vote, v1 ∈ critics ∧ v2 ∈ critics ∧ v1 ≠ v2 ∧ v1.actor = v2.actor ∧ v1.actress = v2.actress :=
sorry

end oscars_voting_problem_l2797_279774


namespace fraction_reduction_l2797_279709

theorem fraction_reduction (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2*a*b) / (a^2 + c^2 - b^2 + 2*a*c) = (a + b - c) / (a - b + c) :=
by sorry

end fraction_reduction_l2797_279709


namespace symmetric_line_l2797_279731

/-- Given a line L1 with equation y = 2x + 1 and a line of symmetry L2 with equation y + 2 = 0,
    the symmetric line L3 has the equation 2x + y + 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (y = 2*x + 1) →  -- Original line L1
  (y = -2)      →  -- Line of symmetry L2 (y + 2 = 0 rearranged)
  (2*x + y + 5 = 0) -- Symmetric line L3
  := by sorry

end symmetric_line_l2797_279731


namespace quadratic_function_range_l2797_279755

/-- Given a quadratic function f(x) = ax^2 - 2ax + c where f(2017) < f(-2016),
    prove that the set of real numbers m satisfying f(m) ≤ f(0) is [0, 2] -/
theorem quadratic_function_range (a c : ℝ) : 
  let f := λ x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) → 
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
sorry

end quadratic_function_range_l2797_279755


namespace workshop_average_salary_l2797_279791

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_avg_salary : ℚ)
  (other_avg_salary : ℚ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : technician_avg_salary = 12000)
  (h4 : other_avg_salary = 6000) :
  (technicians * technician_avg_salary + (total_workers - technicians) * other_avg_salary) / total_workers = 8000 :=
by sorry

end workshop_average_salary_l2797_279791


namespace not_always_greater_quotient_l2797_279757

theorem not_always_greater_quotient : ¬ ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → (∃ n : ℤ, b = n / 10) → a / b > a := by
  sorry

end not_always_greater_quotient_l2797_279757


namespace calculation_proof_l2797_279723

theorem calculation_proof : (0.8 * 60 - 2/5 * 35) * Real.sqrt 144 = 408 := by
  sorry

end calculation_proof_l2797_279723


namespace count_four_digit_with_five_thousands_l2797_279766

/-- A four-digit positive integer with thousands digit 5 -/
def FourDigitWithFiveThousands : Type := { n : ℕ // 5000 ≤ n ∧ n ≤ 5999 }

/-- The count of four-digit positive integers with thousands digit 5 -/
def CountFourDigitWithFiveThousands : ℕ := Finset.card (Finset.filter (λ n => 5000 ≤ n ∧ n ≤ 5999) (Finset.range 10000))

theorem count_four_digit_with_five_thousands :
  CountFourDigitWithFiveThousands = 1000 := by
  sorry

end count_four_digit_with_five_thousands_l2797_279766


namespace imaginary_part_of_z_l2797_279734

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l2797_279734


namespace genevieve_errors_fixed_l2797_279742

/-- Represents a programmer's coding and debugging process -/
structure Programmer where
  total_lines : ℕ
  debug_interval : ℕ
  errors_per_debug : ℕ

/-- Calculates the total number of errors fixed by a programmer -/
def total_errors_fixed (p : Programmer) : ℕ :=
  (p.total_lines / p.debug_interval) * p.errors_per_debug

/-- Theorem stating that under given conditions, the programmer fixes 129 errors -/
theorem genevieve_errors_fixed :
  ∀ (p : Programmer),
    p.total_lines = 4300 →
    p.debug_interval = 100 →
    p.errors_per_debug = 3 →
    total_errors_fixed p = 129 := by
  sorry


end genevieve_errors_fixed_l2797_279742


namespace factor_w4_minus_81_l2797_279775

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) ∧ 
  (∀ (p q : ℝ → ℝ) (a b c : ℝ), (w^4 - 81 = p w * q w ∧ p a = 0 ∧ q b = 0) → 
    (c = 3 ∨ c = -3 ∨ (c^2 = -9 ∧ (∀ x : ℝ, x^2 ≠ -9)))) := by
  sorry

end factor_w4_minus_81_l2797_279775


namespace operation_equivalence_l2797_279761

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equivalence 
  (star mul : Operation) 
  (h_unique : star ≠ mul) 
  (h_eq : (apply_op star 16 4) / (apply_op mul 10 2) = 4) :
  (apply_op star 5 15) / (apply_op mul 8 12) = 30 := by
  sorry


end operation_equivalence_l2797_279761


namespace obtain_a_to_six_l2797_279708

/-- Given a^4 and a^6 - 1, prove that a^6 can be obtained using +, -, and · operations -/
theorem obtain_a_to_six (a : ℝ) : ∃ f : ℝ → ℝ → ℝ → ℝ, 
  f (a^4) (a^6 - 1) 1 = a^6 ∧ 
  (∀ x y z, f x y z = x + y ∨ f x y z = x - y ∨ f x y z = x * y ∨ 
            f x y z = y + z ∨ f x y z = y - z ∨ f x y z = y * z ∨
            f x y z = z + x ∨ f x y z = z - x ∨ f x y z = z * x) :=
by
  sorry

end obtain_a_to_six_l2797_279708


namespace gcd_420_135_l2797_279730

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end gcd_420_135_l2797_279730


namespace complex_magnitude_l2797_279704

theorem complex_magnitude (z : ℂ) : z + 2 * Complex.I = (3 - Complex.I ^ 3) / (1 + Complex.I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end complex_magnitude_l2797_279704


namespace emily_candy_problem_l2797_279746

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := 5

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from her older sister -/
def candy_from_sister : ℕ := 13

theorem emily_candy_problem :
  candy_from_sister = (candy_eaten_per_day * days_candy_lasted) - candy_from_neighbors := by
  sorry

end emily_candy_problem_l2797_279746


namespace smallest_integer_bound_l2797_279710

theorem smallest_integer_bound (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d) / 4 = 74 →
  d = 90 →
  max a (max b c) ≤ d →
  min a (min b c) ≥ 29 :=
by sorry

end smallest_integer_bound_l2797_279710


namespace tan_half_angle_formula_22_5_degrees_l2797_279762

theorem tan_half_angle_formula_22_5_degrees : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end tan_half_angle_formula_22_5_degrees_l2797_279762


namespace median_of_special_arithmetic_sequence_l2797_279756

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d ≠ 0 -- Non-zero common difference
  h2 : ∀ n, a (n + 1) = a n + d -- Arithmetic sequence property
  h3 : a 3 = 8 -- Third term is 8
  h4 : ∃ r, r ≠ 0 ∧ a 1 * r = a 3 ∧ a 3 * r = a 7 -- Geometric sequence property for a₁, a₃, a₇

/-- The median of a 9-term arithmetic sequence with specific properties is 24 -/
theorem median_of_special_arithmetic_sequence (seq : ArithmeticSequence) : 
  seq.a 5 = 24 := by
  sorry

end median_of_special_arithmetic_sequence_l2797_279756


namespace cube_triangle_areas_sum_l2797_279785

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2 × 2 × 2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The main theorem -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 972 := by
  sorry

end cube_triangle_areas_sum_l2797_279785


namespace paths_in_7x8_grid_l2797_279747

/-- The number of paths in a grid with only upward and rightward movements -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid is 6435 -/
theorem paths_in_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end paths_in_7x8_grid_l2797_279747


namespace tank_b_circumference_l2797_279738

/-- The circumference of Tank B given the conditions of the problem -/
theorem tank_b_circumference : 
  ∀ (h_a h_b c_a c_b r_a r_b v_a v_b : ℝ),
  h_a = 7 →
  h_b = 8 →
  c_a = 8 →
  c_a = 2 * Real.pi * r_a →
  v_a = Real.pi * r_a^2 * h_a →
  v_b = Real.pi * r_b^2 * h_b →
  v_a = 0.5600000000000001 * v_b →
  c_b = 2 * Real.pi * r_b →
  c_b = 10 :=
by
  sorry

end tank_b_circumference_l2797_279738


namespace sum_reciprocal_n_n_plus_three_l2797_279700

/-- The sum of the series ∑_{n=1}^∞ 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end sum_reciprocal_n_n_plus_three_l2797_279700


namespace coefficient_x_squared_is_seven_l2797_279784

/-- The coefficient of x^2 in the expansion of (1 - 3x)^7 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 7 6

/-- Theorem: The coefficient of x^2 in the expansion of (1 - 3x)^7 is 7 -/
theorem coefficient_x_squared_is_seven : coefficient_x_squared = 7 := by
  sorry

end coefficient_x_squared_is_seven_l2797_279784


namespace origami_distribution_l2797_279732

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) : 
  total_papers = 48 → 
  num_cousins = 6 → 
  total_papers = num_cousins * papers_per_cousin →
  papers_per_cousin = 8 := by
  sorry

end origami_distribution_l2797_279732


namespace correct_sampling_order_l2797_279726

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the characteristics of each scenario
structure Scenario where
  population_size : ℕ
  has_subgroups : Bool
  has_orderly_numbering : Bool

-- Define the three given scenarios
def scenario1 : Scenario := ⟨8, false, false⟩
def scenario2 : Scenario := ⟨2100, true, false⟩
def scenario3 : Scenario := ⟨700, false, true⟩

-- Function to determine the most appropriate sampling method for a given scenario
def appropriate_method (s : Scenario) : SamplingMethod :=
  if s.population_size ≤ 10 && !s.has_subgroups && !s.has_orderly_numbering then
    SamplingMethod.SimpleRandom
  else if s.has_subgroups then
    SamplingMethod.Stratified
  else if s.has_orderly_numbering then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Theorem stating that the given order of sampling methods is correct for the three scenarios
theorem correct_sampling_order :
  (appropriate_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriate_method scenario2 = SamplingMethod.Stratified) ∧
  (appropriate_method scenario3 = SamplingMethod.Systematic) := by
  sorry

end correct_sampling_order_l2797_279726


namespace sum_of_fourth_and_fifth_terms_l2797_279733

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_and_fifth_terms 
  (a : ℕ → ℕ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 11)
  (h_sixth : a 6 = 43) :
  a 4 + a 5 = 62 :=
by sorry

end sum_of_fourth_and_fifth_terms_l2797_279733


namespace line_tangent_to_circle_and_parabola_l2797_279717

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 2

/-- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l -/
def l (x y : ℝ) : Prop := y = Real.sqrt 2

theorem line_tangent_to_circle_and_parabola :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ l p.1 p.2 ∧
  ∃! q : ℝ × ℝ, C₂ q.1 q.2 ∧ l q.1 q.2 :=
sorry

end line_tangent_to_circle_and_parabola_l2797_279717


namespace three_digit_square_mod_1000_l2797_279711

theorem three_digit_square_mod_1000 (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000] ↔ n = 376 ∨ n = 625) := by
  sorry

end three_digit_square_mod_1000_l2797_279711


namespace quadratic_equation_solution_l2797_279745

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- Proof goes here
  sorry

end quadratic_equation_solution_l2797_279745


namespace dance_step_ratio_l2797_279737

theorem dance_step_ratio : 
  ∀ (N J : ℕ),
  (∃ (k : ℕ), N = k * J) →  -- Nancy steps k times as often as Jason
  N + J = 32 →              -- Total steps
  J = 8 →                   -- Jason's steps
  N / J = 3 :=              -- Ratio of Nancy's to Jason's steps
by sorry

end dance_step_ratio_l2797_279737


namespace no_real_solutions_for_complex_product_l2797_279758

theorem no_real_solutions_for_complex_product : 
  ¬∃ (x : ℝ), (Complex.I : ℂ).im * ((x + 2 + Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 4) + Complex.I)).im = 0 := by
  sorry

end no_real_solutions_for_complex_product_l2797_279758


namespace bricks_needed_for_wall_l2797_279783

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  baseLength : ℝ
  topLength : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the number of bricks needed to build the wall -/
def calculateBricksNeeded (brickDim : BrickDimensions) (wallDim : WallDimensions) (mortarThickness : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of bricks needed for the given wall -/
theorem bricks_needed_for_wall 
  (brickDim : BrickDimensions)
  (wallDim : WallDimensions)
  (mortarThickness : ℝ)
  (h1 : brickDim.length = 125)
  (h2 : brickDim.height = 11.25)
  (h3 : brickDim.thickness = 6)
  (h4 : wallDim.baseLength = 800)
  (h5 : wallDim.topLength = 650)
  (h6 : wallDim.height = 600)
  (h7 : wallDim.thickness = 22.5)
  (h8 : mortarThickness = 1.25) :
  calculateBricksNeeded brickDim wallDim mortarThickness = 1036 :=
sorry

end bricks_needed_for_wall_l2797_279783


namespace eight_liter_solution_exists_l2797_279744

/-- Represents the state of the buckets -/
structure BucketState :=
  (bucket10 : ℕ)
  (bucket6 : ℕ)

/-- Represents a valid operation on the buckets -/
inductive BucketOperation
  | FillFrom10To6
  | FillFrom6To10
  | Empty10
  | Empty6
  | Fill10
  | Fill6

/-- Applies a bucket operation to a given state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillFrom10To6 => 
      let transfer := min state.bucket10 (6 - state.bucket6)
      ⟨state.bucket10 - transfer, state.bucket6 + transfer⟩
  | BucketOperation.FillFrom6To10 => 
      let transfer := min state.bucket6 (10 - state.bucket10)
      ⟨state.bucket10 + transfer, state.bucket6 - transfer⟩
  | BucketOperation.Empty10 => ⟨0, state.bucket6⟩
  | BucketOperation.Empty6 => ⟨state.bucket10, 0⟩
  | BucketOperation.Fill10 => ⟨10, state.bucket6⟩
  | BucketOperation.Fill6 => ⟨state.bucket10, 6⟩

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def checkSolution (ops : List BucketOperation) : Bool :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.bucket10 = 8 ∨ finalState.bucket6 = 8

/-- Theorem: There exists a sequence of operations that results in 8 liters in one bucket -/
theorem eight_liter_solution_exists : ∃ (ops : List BucketOperation), checkSolution ops := by
  sorry


end eight_liter_solution_exists_l2797_279744


namespace initial_bottle_caps_l2797_279793

/-- Given the number of bottle caps lost and the final number of bottle caps,
    calculate the initial number of bottle caps. -/
theorem initial_bottle_caps (lost final : ℝ) : 
  lost = 18.0 → final = 45 → lost + final = 63.0 := by sorry

end initial_bottle_caps_l2797_279793


namespace matrix_equation_holds_l2797_279768

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_holds :
  let s : ℤ := -2
  let t : ℤ := -6
  let u : ℤ := -14
  let v : ℤ := -13
  A^4 + s • A^3 + t • A^2 + u • A + v • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) := by
  sorry

end matrix_equation_holds_l2797_279768


namespace sean_charles_whistle_difference_l2797_279781

theorem sean_charles_whistle_difference : 
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 223 →
    charles_whistles = 128 →
    sean_whistles - charles_whistles = 95 := by
  sorry

end sean_charles_whistle_difference_l2797_279781


namespace arithmetic_sequence_sum_l2797_279776

/-- Given an arithmetic sequence {aₙ} with S₁ = 10 and S₂ = 20, prove that S₁₀ = 100 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  S 1 = 10 →                            -- given S₁ = 10
  S 2 = 20 →                            -- given S₂ = 20
  S 10 = 100 := by                      -- prove S₁₀ = 100
sorry


end arithmetic_sequence_sum_l2797_279776


namespace inequality_proof_l2797_279749

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
  sorry

end inequality_proof_l2797_279749


namespace apple_cost_calculation_l2797_279715

def total_budget : ℝ := 60
def hummus_cost : ℝ := 5
def hummus_quantity : ℕ := 2
def chicken_cost : ℝ := 20
def bacon_cost : ℝ := 10
def vegetables_cost : ℝ := 10
def apple_quantity : ℕ := 5

theorem apple_cost_calculation :
  let other_items_cost := hummus_cost * hummus_quantity + chicken_cost + bacon_cost + vegetables_cost
  let remaining_budget := total_budget - other_items_cost
  remaining_budget / apple_quantity = 2 := by sorry

end apple_cost_calculation_l2797_279715


namespace vectors_problem_l2797_279753

def a : ℝ × ℝ := (3, -1)
def b (k : ℝ) : ℝ × ℝ := (1, k)

theorem vectors_problem (k : ℝ) 
  (h : a.1 * (b k).1 + a.2 * (b k).2 = 0) : 
  k = 3 ∧ 
  (a.1 + (b k).1) * (a.1 - (b k).1) + (a.2 + (b k).2) * (a.2 - (b k).2) = 0 := by
  sorry

end vectors_problem_l2797_279753


namespace equation_one_solution_l2797_279718

theorem equation_one_solution : 
  {x : ℝ | (x + 3)^2 - 9 = 0} = {0, -6} := by sorry

end equation_one_solution_l2797_279718


namespace greatest_area_difference_l2797_279794

/-- A rectangle with integer dimensions and perimeter 200 cm -/
structure Rectangle where
  width : ℕ
  height : ℕ
  perimeter_eq : width * 2 + height * 2 = 200

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A rectangle with one side of length 80 cm -/
structure DoorRectangle extends Rectangle where
  door_side : width = 80 ∨ height = 80

theorem greatest_area_difference :
  ∃ (r : Rectangle) (d : DoorRectangle),
    ∀ (r' : Rectangle) (d' : DoorRectangle),
      d.area - r.area ≥ d'.area - r'.area ∧
      d.area - r.area = 2300 := by
  sorry

end greatest_area_difference_l2797_279794


namespace min_value_sum_reciprocals_l2797_279786

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end min_value_sum_reciprocals_l2797_279786
