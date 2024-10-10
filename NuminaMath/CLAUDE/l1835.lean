import Mathlib

namespace flour_to_baking_soda_ratio_l1835_183547

/-- Prove that the ratio of flour to baking soda is 10 to 1 given the specified conditions -/
theorem flour_to_baking_soda_ratio 
  (sugar : ℕ) 
  (flour : ℕ) 
  (baking_soda : ℕ) 
  (h1 : sugar * 6 = flour * 5) 
  (h2 : sugar = 2000) 
  (h3 : flour = 8 * (baking_soda + 60)) : 
  flour / baking_soda = 10 := by
  sorry

end flour_to_baking_soda_ratio_l1835_183547


namespace rectangle_dimension_change_l1835_183585

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.4 * L) (h2 : B' * L' = 1.05 * B * L) :
  B' = 0.75 * B :=
sorry

end rectangle_dimension_change_l1835_183585


namespace trigonometric_equation_solution_l1835_183584

theorem trigonometric_equation_solution (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos (5 * x) + 2 * Real.cos (4 * x) + 2 * Real.cos (3 * x) +
   2 * Real.cos (2 * x) + 2 * Real.cos x + 1 = 0) ↔
  (∃ k : ℕ, k ∈ Finset.range 10 ∧ x = 2 * k * Real.pi / 11) :=
by sorry

end trigonometric_equation_solution_l1835_183584


namespace value_of_P_l1835_183577

theorem value_of_P (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 10) : 
  x + y = 4 := by
sorry

end value_of_P_l1835_183577


namespace total_spent_is_108_l1835_183534

/-- The total amount spent by Robert and Teddy on snacks -/
def total_spent (pizza_boxes : ℕ) (pizza_price : ℕ) (robert_drinks : ℕ) (drink_price : ℕ)
                (hamburgers : ℕ) (hamburger_price : ℕ) (teddy_drinks : ℕ) : ℕ :=
  pizza_boxes * pizza_price + robert_drinks * drink_price +
  hamburgers * hamburger_price + teddy_drinks * drink_price

/-- Theorem stating that the total amount spent is $108 -/
theorem total_spent_is_108 :
  total_spent 5 10 10 2 6 3 10 = 108 := by
  sorry

end total_spent_is_108_l1835_183534


namespace quadratic_function_range_l1835_183516

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  f 0 = 3 ∧ 
  f 2 = 1

/-- The range of m for which the function has max 3 and min 1 on [0,m] -/
def ValidRange (f : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1)

/-- The main theorem -/
theorem quadratic_function_range (f : ℝ → ℝ) (h : QuadraticFunction f) :
  {m | ValidRange f m} = Set.Icc 2 4 := by
  sorry

end quadratic_function_range_l1835_183516


namespace tan_2x_and_sin_x_plus_pi_4_l1835_183549

theorem tan_2x_and_sin_x_plus_pi_4 (x : ℝ) 
  (h1 : |Real.tan x| = 2) 
  (h2 : x ∈ Set.Ioo (π / 2) π) : 
  Real.tan (2 * x) = 4 / 3 ∧ 
  Real.sin (x + π / 4) = Real.sqrt 10 / 10 := by
  sorry

end tan_2x_and_sin_x_plus_pi_4_l1835_183549


namespace pure_imaginary_complex_number_l1835_183599

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 :=
by
  sorry

end pure_imaginary_complex_number_l1835_183599


namespace correct_calculation_l1835_183582

theorem correct_calculation (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end correct_calculation_l1835_183582


namespace summer_mowing_times_l1835_183575

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowing : ℕ := 5

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowing : ℕ := 8

/-- The difference between spring and summer mowing times -/
def mowing_difference : ℕ := 3

theorem summer_mowing_times : 
  spring_mowing - summer_mowing = mowing_difference := by sorry

end summer_mowing_times_l1835_183575


namespace max_stamps_purchasable_l1835_183512

theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 25 → budget = 5000 → 
  ∃ n : ℕ, n * stamp_price ≤ budget ∧ 
  ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n ∧ 
  n = 200 :=
by sorry

end max_stamps_purchasable_l1835_183512


namespace cube_sphere_volume_ratio_l1835_183501

theorem cube_sphere_volume_ratio (a r : ℝ) (h : a > 0) (k : r > 0) :
  6 * a^2 = 4 * Real.pi * r^2 →
  (a^3) / ((4/3) * Real.pi * r^3) = Real.sqrt 6 / 6 := by
sorry

end cube_sphere_volume_ratio_l1835_183501


namespace andrew_payment_l1835_183540

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_price grape_quantity mango_price mango_quantity : ℕ) : ℕ :=
  grape_price * grape_quantity + mango_price * mango_quantity

/-- Theorem: Andrew paid 975 to the shopkeeper -/
theorem andrew_payment : total_amount 74 6 59 9 = 975 := by
  sorry

end andrew_payment_l1835_183540


namespace problem_solution_l1835_183500

theorem problem_solution (a : ℚ) : a + a/3 + a/4 = 11/4 → a = 33/19 := by
  sorry

end problem_solution_l1835_183500


namespace frog_final_position_l1835_183522

-- Define the circle points
inductive CirclePoint
| One
| Two
| Three
| Four
| Five

-- Define the jump function
def jump (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.One => CirclePoint.Two
  | CirclePoint.Two => CirclePoint.Four
  | CirclePoint.Three => CirclePoint.Four
  | CirclePoint.Four => CirclePoint.One
  | CirclePoint.Five => CirclePoint.One

-- Define the function to perform multiple jumps
def multiJump (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | Nat.succ m => jump (multiJump start m)

-- Theorem statement
theorem frog_final_position :
  multiJump CirclePoint.Five 1995 = CirclePoint.Four := by
  sorry

end frog_final_position_l1835_183522


namespace large_cube_pieces_l1835_183574

/-- The number of wire pieces needed for a cube framework -/
def wire_pieces (n : ℕ) : ℕ := 3 * (n + 1)^2 * n

/-- The fact that a 2 × 2 × 2 cube uses 54 wire pieces -/
axiom small_cube_pieces : wire_pieces 2 = 54

/-- Theorem: The number of wire pieces needed for a 10 × 10 × 10 cube is 3630 -/
theorem large_cube_pieces : wire_pieces 10 = 3630 := by
  sorry

end large_cube_pieces_l1835_183574


namespace three_buildings_height_l1835_183597

/-- The height of three buildings given specific conditions -/
theorem three_buildings_height 
  (h1 : ℕ) -- Height of the first building
  (h2_eq : h2 = 2 * h1) -- Second building is twice as tall as the first
  (h3_eq : h3 = 3 * (h1 + h2)) -- Third building is three times as tall as the first two combined
  (h1_val : h1 = 600) -- First building is 600 feet tall
  : h1 + h2 + h3 = 7200 := by
  sorry

#check three_buildings_height

end three_buildings_height_l1835_183597


namespace circle_intersection_chord_length_l1835_183598

/-- A circle in the xy-plane -/
structure Circle where
  a : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y ↦ x^2 + y^2 + 2*x - 2*y + a = 0

/-- A line in the xy-plane -/
def Line : ℝ → ℝ → Prop :=
  fun x y ↦ x + y + 2 = 0

/-- The length of a chord formed by the intersection of a circle and a line -/
def ChordLength (c : Circle) : ℝ :=
  4 -- Given in the problem

/-- The main theorem -/
theorem circle_intersection_chord_length (c : Circle) :
  (∀ x y, Line x y → c.equation x y) →
  ChordLength c = 4 →
  c.a = -4 := by
  sorry

end circle_intersection_chord_length_l1835_183598


namespace distance_between_ports_l1835_183578

/-- The distance between ports A and B in kilometers -/
def distance_AB : ℝ := 40

/-- The speed of the ship in still water in km/h -/
def ship_speed : ℝ := 26

/-- The speed of the river current in km/h -/
def current_speed : ℝ := 6

/-- The number of round trips made by the ship -/
def round_trips : ℕ := 4

/-- The total time taken for all round trips in hours -/
def total_time : ℝ := 13

theorem distance_between_ports :
  let downstream_speed := ship_speed + current_speed
  let upstream_speed := ship_speed - current_speed
  let time_per_round_trip := total_time / round_trips
  let downstream_time := (upstream_speed * time_per_round_trip) / (downstream_speed + upstream_speed)
  distance_AB = downstream_speed * downstream_time :=
by sorry

end distance_between_ports_l1835_183578


namespace quadratic_max_l1835_183557

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem quadratic_max :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 2) :=
sorry

end quadratic_max_l1835_183557


namespace calculation_result_l1835_183542

theorem calculation_result : (25 * 8 + 1 / (5/7)) / (2014 - 201.4 * 2) = 1/8 := by
  sorry

end calculation_result_l1835_183542


namespace k_increasing_range_l1835_183576

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the domain of f
def D : Set ℝ := { x | x ≥ -1 }

-- Define the property of being k-increasing on a set
def is_k_increasing (f : ℝ → ℝ) (k : ℝ) (S : Set ℝ) : Prop :=
  k ≠ 0 ∧ ∀ x ∈ S, (x + k) ∈ S → f (x + k) ≥ f x

-- State the theorem
theorem k_increasing_range (k : ℝ) :
  is_k_increasing f k D → k ≥ 2 := by sorry

end k_increasing_range_l1835_183576


namespace cyclic_sum_inequality_l1835_183566

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
  sorry

end cyclic_sum_inequality_l1835_183566


namespace minimum_sum_of_parameters_l1835_183535

theorem minimum_sum_of_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b = 1) → (a + b ≥ 4) ∧ (∃ a b, 1 / a + 1 / b = 1 ∧ a + b = 4) :=
sorry

end minimum_sum_of_parameters_l1835_183535


namespace mn_max_and_m2n2_min_l1835_183586

/-- Given real numbers m and n, where m > 0, n > 0, and 2m + n = 1,
    prove that the maximum value of mn is 1/8 and
    the minimum value of 4m^2 + n^2 is 1/2 -/
theorem mn_max_and_m2n2_min (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → m * n ≥ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  m * n = 1/8 ∧ 4 * m^2 + n^2 = 1/2 := by
  sorry

end mn_max_and_m2n2_min_l1835_183586


namespace lee_makes_27_cookies_l1835_183538

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  18 * cups / 2

/-- Theorem stating that Lee can make 27 cookies with 3 cups of flour. -/
theorem lee_makes_27_cookies : cookies_from_flour 3 = 27 := by
  sorry

end lee_makes_27_cookies_l1835_183538


namespace magnitude_of_vector_combination_l1835_183580

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- State the theorem
theorem magnitude_of_vector_combination :
  ‖(3 • a) - b‖ = 4 * Real.sqrt 2 := by
  sorry

end magnitude_of_vector_combination_l1835_183580


namespace shaded_region_perimeter_l1835_183546

structure Grid :=
  (size : Nat)
  (shaded : List (Nat × Nat))

def isExternal (g : Grid) (pos : Nat × Nat) : Bool :=
  let (x, y) := pos
  x = 1 ∨ x = g.size ∨ y = 1 ∨ y = g.size

def countExternalEdges (g : Grid) : Nat :=
  g.shaded.foldl (fun acc pos =>
    acc + (if isExternal g pos then
             (if pos.1 = 1 then 1 else 0) +
             (if pos.1 = g.size then 1 else 0) +
             (if pos.2 = 1 then 1 else 0) +
             (if pos.2 = g.size then 1 else 0)
           else 0)
  ) 0

theorem shaded_region_perimeter (g : Grid) :
  g.size = 3 ∧
  g.shaded = [(1,2), (2,1), (2,3), (3,2)] →
  countExternalEdges g = 10 := by
  sorry

end shaded_region_perimeter_l1835_183546


namespace katie_speed_calculation_l1835_183507

-- Define the running speeds
def eugene_speed : ℚ := 4
def brianna_speed : ℚ := (2/3) * eugene_speed
def katie_speed : ℚ := (7/5) * brianna_speed

-- Theorem to prove
theorem katie_speed_calculation :
  katie_speed = 56/15 := by sorry

end katie_speed_calculation_l1835_183507


namespace greatest_integer_with_gcf_five_exists_185_is_greatest_l1835_183520

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem is_greatest : ∀ m : ℕ, m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ 185 :=
by
  sorry

end greatest_integer_with_gcf_five_exists_185_is_greatest_l1835_183520


namespace max_added_value_l1835_183528

/-- The added value function for the car manufacturer's production line renovation --/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- The theorem stating the maximum value of the added value function --/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4*a/5) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo 0 (4*a/5) → f a y ≤ f a x) ∧
    f a x = 32 * a^3 / 27 ∧
    x = 2*a/3 := by
  sorry

end max_added_value_l1835_183528


namespace correct_sampling_methods_l1835_183593

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics -/
structure Survey where
  total_population : ℕ
  strata : List ℕ
  sample_size : ℕ

/-- Determines the appropriate sampling method for a given survey -/
def appropriate_sampling_method (s : Survey) : SamplingMethod :=
  if s.strata.length > 1 then SamplingMethod.Stratified else SamplingMethod.Random

/-- The two surveys from the problem -/
def survey1 : Survey :=
  { total_population := 500
  , strata := [125, 280, 95]
  , sample_size := 100 }

def survey2 : Survey :=
  { total_population := 12
  , strata := [12]
  , sample_size := 3 }

/-- Theorem stating the correct sampling methods for the given surveys -/
theorem correct_sampling_methods :
  (appropriate_sampling_method survey1 = SamplingMethod.Stratified) ∧
  (appropriate_sampling_method survey2 = SamplingMethod.Random) := by
  sorry

end correct_sampling_methods_l1835_183593


namespace sum_of_primes_divisible_by_12_l1835_183550

theorem sum_of_primes_divisible_by_12 (p q : ℕ) : 
  Prime p → Prime q → p - q = 2 → q > 3 → ∃ k : ℕ, p + q = 12 * k := by
sorry

end sum_of_primes_divisible_by_12_l1835_183550


namespace matrix_determinant_l1835_183526

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
  sorry

end matrix_determinant_l1835_183526


namespace product_primitive_roots_congruent_one_l1835_183544

/-- Given a prime p > 3, the product of all primitive roots modulo p is congruent to 1 modulo p -/
theorem product_primitive_roots_congruent_one (p : Nat) (hp : p.Prime) (hp3 : p > 3) :
  ∃ (S : Finset Nat), 
    (∀ s ∈ S, 1 ≤ s ∧ s < p ∧ IsPrimitiveRoot s p) ∧ 
    (∀ x, 1 ≤ x ∧ x < p ∧ IsPrimitiveRoot x p → x ∈ S) ∧
    (S.prod id) % p = 1 := by
  sorry


end product_primitive_roots_congruent_one_l1835_183544


namespace algebraic_simplification_l1835_183533

theorem algebraic_simplification (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b := by
  sorry

end algebraic_simplification_l1835_183533


namespace monster_hunt_proof_l1835_183556

/-- The sum of a geometric sequence with initial term 2, common ratio 2, and 5 terms -/
def monster_sum : ℕ := 
  List.range 5
  |> List.map (fun n => 2 * 2^n)
  |> List.sum

theorem monster_hunt_proof : monster_sum = 62 := by
  sorry

end monster_hunt_proof_l1835_183556


namespace tangent_point_coordinates_l1835_183587

/-- A point on the curve y = x^3 - 3x with a tangent line parallel to the x-axis -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 3*x
  parallel_tangent : 3*x^2 - 3 = 0

theorem tangent_point_coordinates (P : TangentPoint) : 
  (P.x = 1 ∧ P.y = -2) ∨ (P.x = -1 ∧ P.y = 2) := by
  sorry

end tangent_point_coordinates_l1835_183587


namespace floor_equation_difference_l1835_183506

theorem floor_equation_difference : ∃ (x y : ℤ), 
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = 102 → z ≤ x) ∧ 
  (⌊(x : ℚ) / 3⌋ = 102) ∧
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = -102 → y ≤ z) ∧ 
  (⌊(y : ℚ) / 3⌋ = -102) ∧
  (x - y = 614) := by
sorry

end floor_equation_difference_l1835_183506


namespace apple_lovers_l1835_183594

structure FruitPreferences where
  total : ℕ
  apple : ℕ
  orange : ℕ
  mango : ℕ
  banana : ℕ
  grapes : ℕ
  orange_mango_not_apple : ℕ
  mango_apple_not_orange : ℕ
  all_three : ℕ
  banana_grapes_only : ℕ
  apple_banana_grapes_not_others : ℕ

def room : FruitPreferences := {
  total := 60,
  apple := 40,
  orange := 17,
  mango := 23,
  banana := 12,
  grapes := 9,
  orange_mango_not_apple := 7,
  mango_apple_not_orange := 10,
  all_three := 4,
  banana_grapes_only := 6,
  apple_banana_grapes_not_others := 3
}

theorem apple_lovers (pref : FruitPreferences) : pref.apple = 40 :=
  sorry

end apple_lovers_l1835_183594


namespace maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l1835_183591

/-- The set of reciprocals of natural numbers -/
def S : Set ℚ := {q : ℚ | ∃ n : ℕ, q = 1 / n}

/-- An arithmetic progression in S -/
def is_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  ∃ (first d : ℚ), ∀ i < n, a i = first + i • d ∧ a i ∈ S

/-- A maximal arithmetic progression in S -/
def is_maximal_arithmetic_progression (a : ℕ → ℚ) (n : ℕ) : Prop :=
  is_arithmetic_progression a n ∧
  ¬∃ (b : ℕ → ℚ) (m : ℕ), m > n ∧ is_arithmetic_progression b m ∧
    (∀ i < n, a i = b i)

theorem maximal_arithmetic_progression_1996 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1996 :=
sorry

theorem maximal_arithmetic_progression_1997 :
  ∃ (a : ℕ → ℚ), is_maximal_arithmetic_progression a 1997 :=
sorry

end maximal_arithmetic_progression_1996_maximal_arithmetic_progression_1997_l1835_183591


namespace z_range_in_parallelogram_l1835_183548

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)
def D : ℝ × ℝ := (4, 0)

-- Define a function to check if a point is within or on the boundary of the parallelogram
def isInParallelogram (p : ℝ × ℝ) : Prop := sorry

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := 2 * p.1 - 5 * p.2

-- State the theorem
theorem z_range_in_parallelogram :
  ∀ p : ℝ × ℝ, isInParallelogram p → -14 ≤ z p ∧ z p ≤ 18 := by sorry

end z_range_in_parallelogram_l1835_183548


namespace terry_bottle_caps_l1835_183508

def bottle_cap_collection (num_groups : ℕ) (caps_per_group : ℕ) : ℕ :=
  num_groups * caps_per_group

theorem terry_bottle_caps : 
  bottle_cap_collection 80 7 = 560 := by
  sorry

end terry_bottle_caps_l1835_183508


namespace right_triangle_shorter_leg_l1835_183509

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16 units long
by sorry

end right_triangle_shorter_leg_l1835_183509


namespace correct_green_pens_l1835_183588

-- Define the ratio of blue pens to green pens
def blue_to_green_ratio : ℚ := 5 / 3

-- Define the number of blue pens
def blue_pens : ℕ := 20

-- Define the number of green pens
def green_pens : ℕ := 12

-- Theorem to prove
theorem correct_green_pens : 
  (blue_pens : ℚ) / green_pens = blue_to_green_ratio := by
  sorry

end correct_green_pens_l1835_183588


namespace temperature_range_l1835_183589

/-- Given the highest and lowest temperatures on a certain day, 
    prove that the range of temperature change is between these two values, inclusive. -/
theorem temperature_range (highest lowest t : ℝ) 
  (h_highest : highest = 26) 
  (h_lowest : lowest = 12) 
  (h_range : lowest ≤ t ∧ t ≤ highest) : 
  12 ≤ t ∧ t ≤ 26 := by
  sorry

end temperature_range_l1835_183589


namespace lindas_additional_dimes_l1835_183505

/-- The number of additional dimes Linda's mother gives her -/
def additional_dimes : ℕ := 2

/-- The initial number of dimes Linda has -/
def initial_dimes : ℕ := 2

/-- The initial number of quarters Linda has -/
def initial_quarters : ℕ := 6

/-- The initial number of nickels Linda has -/
def initial_nickels : ℕ := 5

/-- The number of additional quarters Linda's mother gives her -/
def additional_quarters : ℕ := 10

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

theorem lindas_additional_dimes :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end lindas_additional_dimes_l1835_183505


namespace some_number_value_l1835_183573

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 9 := by
  sorry

end some_number_value_l1835_183573


namespace fraction_simplification_l1835_183570

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end fraction_simplification_l1835_183570


namespace min_a_value_l1835_183564

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the relationship between f, g, and 2^x
axiom f_g_sum : ∀ x ∈ Set.Icc 1 2, f x + g x = 2^x

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, a * f x + g (2*x) ≥ 0

-- State the theorem
theorem min_a_value :
  ∃ a_min : ℝ, a_min = -17/6 ∧
  (∀ a, inequality_holds a ↔ a ≥ a_min) :=
sorry

end min_a_value_l1835_183564


namespace doubled_factorial_30_trailing_zeros_l1835_183583

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in 2 * n! -/
def trailingZerosDoubled (n : ℕ) : ℕ := sorry

theorem doubled_factorial_30_trailing_zeros :
  trailingZerosDoubled 30 = 7 := by sorry

end doubled_factorial_30_trailing_zeros_l1835_183583


namespace min_stamps_for_30_cents_l1835_183531

/-- Represents the number of stamps needed to make a certain value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def value (s : StampCombination) : ℕ := 3 * s.threes + 4 * s.fours

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ := s.threes + s.fours

/-- Checks if a StampCombination is valid for the given target value -/
def is_valid (s : StampCombination) (target : ℕ) : Prop :=
  value s = target

/-- Theorem: The minimum number of stamps needed to make 30 cents is 8 -/
theorem min_stamps_for_30_cents :
  ∃ (s : StampCombination), is_valid s 30 ∧
    total_stamps s = 8 ∧
    (∀ (t : StampCombination), is_valid t 30 → total_stamps s ≤ total_stamps t) :=
sorry

end min_stamps_for_30_cents_l1835_183531


namespace least_number_to_add_l1835_183502

def problem (x : ℕ) : Prop :=
  let lcm := 7 * 11 * 13 * 17 * 19
  (∃ k : ℕ, (625573 + x) = k * lcm) ∧
  (∀ y : ℕ, y < x → ¬∃ k : ℕ, (625573 + y) = k * lcm)

theorem least_number_to_add : problem 21073 := by
  sorry

end least_number_to_add_l1835_183502


namespace mn_inequality_l1835_183521

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set P
def P : Set ℝ := {x | f x > 4}

-- State the theorem
theorem mn_inequality (m n : ℝ) (hm : m ∈ P) (hn : n ∈ P) :
  |m * n + 4| > 2 * |m + n| := by
  sorry

end mn_inequality_l1835_183521


namespace square_minus_one_divisible_by_three_l1835_183581

theorem square_minus_one_divisible_by_three (n : ℕ) (h : ¬ 3 ∣ n) : 3 ∣ (n^2 - 1) :=
by sorry

end square_minus_one_divisible_by_three_l1835_183581


namespace arithmetic_sequence_sum_l1835_183555

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 6 + a 11 = 3 →
  a 3 + a 9 = 2 := by
sorry

end arithmetic_sequence_sum_l1835_183555


namespace least_five_digit_square_cube_l1835_183529

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    (m < 10000 ∨ m ≥ 100000) ∨
    (∀ a : ℕ, m ≠ a^2) ∨
    (∀ b : ℕ, m ≠ b^3)) ∧
  n = 15625 :=
by sorry

end least_five_digit_square_cube_l1835_183529


namespace eighth_root_of_5487587353601_l1835_183532

theorem eighth_root_of_5487587353601 : ∃ n : ℕ, n ^ 8 = 5487587353601 ∧ n = 101 := by
  sorry

end eighth_root_of_5487587353601_l1835_183532


namespace solution_count_theorem_l1835_183558

/-- The number of solutions to the equation 2x + 3y + z + x^2 = n for positive integers x, y, z -/
def num_solutions (n : ℕ+) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    2 * t.1 + 3 * t.2.1 + t.2.2 + t.1 * t.1 = n.val ∧ 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0) (Finset.product (Finset.range n.val) (Finset.product (Finset.range n.val) (Finset.range n.val)))).card

theorem solution_count_theorem (n : ℕ+) : 
  num_solutions n = 25 → n = 32 ∨ n = 33 := by
  sorry

end solution_count_theorem_l1835_183558


namespace special_triangle_smallest_angle_cos_l1835_183565

/-- A triangle with sides of three consecutive odd numbers where the largest angle is thrice the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n + 2
  side2 : ℕ := n + 3
  side3 : ℕ := n + 4
  is_valid : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1
  largest_angle_triple : Real.cos ((n + 1) / (2 * (n + 2))) = 
    4 * ((n + 5) / (2 * (n + 4))) ^ 3 - 3 * ((n + 5) / (2 * (n + 4)))

/-- The cosine of the smallest angle in a SpecialTriangle is 6/11 -/
theorem special_triangle_smallest_angle_cos (t : SpecialTriangle) : 
  Real.cos ((t.n + 5) / (2 * (t.n + 4))) = 6 / 11 := by
  sorry

end special_triangle_smallest_angle_cos_l1835_183565


namespace debate_team_formations_l1835_183592

def num_boys : ℕ := 3
def num_girls : ℕ := 3
def num_debaters : ℕ := 4
def boy_a_exists : Prop := true

theorem debate_team_formations :
  (num_boys + num_girls - 1) * (num_boys + num_girls - 1) * (num_boys + num_girls - 2) * (num_boys + num_girls - 3) = 300 :=
by sorry

end debate_team_formations_l1835_183592


namespace quadratic_two_distinct_real_roots_l1835_183563

theorem quadratic_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l1835_183563


namespace andrew_worked_300_days_l1835_183519

/-- Represents the company's vacation policy and Andrew's vacation usage --/
structure VacationData where
  /-- The number of work days required to earn one vacation day --/
  work_days_per_vacation_day : ℕ
  /-- Vacation days taken in March --/
  march_vacation : ℕ
  /-- Vacation days taken in September --/
  september_vacation : ℕ
  /-- Remaining vacation days --/
  remaining_vacation : ℕ

/-- Calculates the total number of days worked given the vacation data --/
def days_worked (data : VacationData) : ℕ :=
  sorry

/-- Theorem stating that given the specific vacation data, Andrew worked 300 days --/
theorem andrew_worked_300_days : 
  let data : VacationData := {
    work_days_per_vacation_day := 10,
    march_vacation := 5,
    september_vacation := 10,
    remaining_vacation := 15
  }
  days_worked data = 300 := by
  sorry

end andrew_worked_300_days_l1835_183519


namespace power_comparison_l1835_183596

theorem power_comparison : 3^15 < 10^9 ∧ 10^9 < 5^13 := by
  sorry

end power_comparison_l1835_183596


namespace problem_statement_l1835_183571

/-- Given M = 6021 ÷ 4, N = 2M, and X = N - M + 500, prove that X = 3005.25 -/
theorem problem_statement (M N X : ℚ) 
  (hM : M = 6021 / 4)
  (hN : N = 2 * M)
  (hX : X = N - M + 500) :
  X = 3005.25 := by
sorry

end problem_statement_l1835_183571


namespace uncle_ben_chickens_l1835_183553

/-- Represents Uncle Ben's farm --/
structure Farm where
  roosters : Nat
  nonLayingHens : Nat
  eggsPerLayingHen : Nat
  totalEggs : Nat

/-- Calculates the total number of chickens on the farm --/
def totalChickens (f : Farm) : Nat :=
  let layingHens := f.totalEggs / f.eggsPerLayingHen
  f.roosters + f.nonLayingHens + layingHens

/-- Theorem stating that Uncle Ben has 440 chickens --/
theorem uncle_ben_chickens :
  ∀ (f : Farm),
    f.roosters = 39 →
    f.nonLayingHens = 15 →
    f.eggsPerLayingHen = 3 →
    f.totalEggs = 1158 →
    totalChickens f = 440 := by
  sorry

end uncle_ben_chickens_l1835_183553


namespace intersection_complement_when_a_2_range_of_a_for_proper_superset_l1835_183543

-- Define sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3*a - 10 ≤ x ∧ x < 2*a + 1}
def Q : Set ℝ := {x | |2*x - 3| ≤ 7}

-- Part 1
theorem intersection_complement_when_a_2 : 
  P 2 ∩ (Set.univ \ Q) = {x | -4 ≤ x ∧ x < -2} := by sorry

-- Part 2
theorem range_of_a_for_proper_superset : 
  {a : ℝ | P a ⊃ Q ∧ P a ≠ Q} = Set.Ioo 2 (8/3) := by sorry

end intersection_complement_when_a_2_range_of_a_for_proper_superset_l1835_183543


namespace gerald_price_is_264_60_verify_hendricks_price_l1835_183513

-- Define the original price of the guitar
def original_price : ℝ := 280

-- Define Hendricks' discount rate
def hendricks_discount_rate : ℝ := 0.15

-- Define Gerald's discount rate
def gerald_discount_rate : ℝ := 0.10

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.05

-- Define Hendricks' final price
def hendricks_price : ℝ := 250

-- Function to calculate the final price after discount and tax
def calculate_final_price (price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem stating that Gerald's price is $264.60
theorem gerald_price_is_264_60 :
  calculate_final_price original_price gerald_discount_rate sales_tax_rate = 264.60 := by
  sorry

-- Theorem verifying Hendricks' price
theorem verify_hendricks_price :
  calculate_final_price original_price hendricks_discount_rate sales_tax_rate = hendricks_price := by
  sorry

end gerald_price_is_264_60_verify_hendricks_price_l1835_183513


namespace remainder_of_98_times_102_mod_9_l1835_183527

theorem remainder_of_98_times_102_mod_9 : (98 * 102) % 9 = 8 := by
  sorry

end remainder_of_98_times_102_mod_9_l1835_183527


namespace isosceles_right_triangle_hypotenuse_l1835_183559

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → 
  c > 0 → 
  c^2 = 2 * a^2 →  -- isosceles right triangle condition
  a^2 + a^2 + c^2 = 1452 →  -- sum of squares condition
  c = Real.sqrt 726 := by
  sorry

end isosceles_right_triangle_hypotenuse_l1835_183559


namespace matrix_determinant_l1835_183562

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3/2; 2, 6]
  Matrix.det A = 27 := by
sorry

end matrix_determinant_l1835_183562


namespace custom_deck_probability_l1835_183552

theorem custom_deck_probability : 
  let total_cards : ℕ := 65
  let spades : ℕ := 14
  let other_suits : ℕ := 13
  let aces : ℕ := 4
  let kings : ℕ := 4
  (aces : ℚ) / total_cards * kings / (total_cards - 1) = 1 / 260 :=
by sorry

end custom_deck_probability_l1835_183552


namespace interest_rate_is_ten_percent_l1835_183595

/-- The interest rate at which A lent money to B, given the conditions of the problem -/
def interest_rate_A_to_B (principal : ℚ) (rate_B_to_C : ℚ) (time : ℚ) (B_gain : ℚ) : ℚ :=
  let interest_from_C := principal * rate_B_to_C * time
  let interest_to_A := interest_from_C - B_gain
  (interest_to_A / (principal * time)) * 100

/-- Theorem stating that the interest rate from A to B is 10% under the given conditions -/
theorem interest_rate_is_ten_percent :
  interest_rate_A_to_B 3500 0.13 3 315 = 10 := by sorry

end interest_rate_is_ten_percent_l1835_183595


namespace divisible_by_sixteen_l1835_183545

theorem divisible_by_sixteen (m n : ℤ) : ∃ k : ℤ, (5*m + 3*n + 1)^5 * (3*m + n + 4)^4 = 16*k := by
  sorry

end divisible_by_sixteen_l1835_183545


namespace power_of_fraction_l1835_183514

theorem power_of_fraction : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end power_of_fraction_l1835_183514


namespace simplify_expression_1_simplify_expression_2_l1835_183518

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

-- Part 2
theorem simplify_expression_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 3) :
  ((x + 3) / x - 2) / ((x^2 - 9) / (4 * x)) = -4 / (x + 3) := by
  sorry

end simplify_expression_1_simplify_expression_2_l1835_183518


namespace special_function_properties_l1835_183515

/-- A function satisfying f(ab) = af(b) + bf(a) for all a, b ∈ ℝ -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = a * f b + b * f a

theorem special_function_properties (f : ℝ → ℝ) 
  (h : special_function f) (h_not_zero : ∃ x, f x ≠ 0) :
  (f 0 = 0 ∧ f 1 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end special_function_properties_l1835_183515


namespace first_term_to_common_difference_ratio_l1835_183530

/-- An arithmetic progression where the sum of the first 14 terms is three times the sum of the first 7 terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (14 * a + 91 * d) = 3 * (7 * a + 21 * d)

/-- The ratio of the first term to the common difference is 4:1 -/
theorem first_term_to_common_difference_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 4 := by
  sorry

end first_term_to_common_difference_ratio_l1835_183530


namespace majorB_higher_admission_rate_male_higher_admission_rate_l1835_183503

/-- Represents the gender of applicants -/
inductive Gender
| Male
| Female

/-- Represents the major of applicants -/
inductive Major
| A
| B

/-- Data structure for application and admission information -/
structure MajorData where
  applicants : Gender → ℕ
  admissionRate : Gender → ℚ

/-- Calculate the weighted average admission rate for a major -/
def weightedAverageAdmissionRate (data : MajorData) : ℚ :=
  let totalApplicants := data.applicants Gender.Male + data.applicants Gender.Female
  let weightedSum := (data.applicants Gender.Male * data.admissionRate Gender.Male) +
                     (data.applicants Gender.Female * data.admissionRate Gender.Female)
  weightedSum / totalApplicants

/-- Calculate the overall admission rate for a gender across both majors -/
def overallAdmissionRate (majorA : MajorData) (majorB : MajorData) (gender : Gender) : ℚ :=
  let totalApplicants := majorA.applicants gender + majorB.applicants gender
  let admittedA := majorA.applicants gender * majorA.admissionRate gender
  let admittedB := majorB.applicants gender * majorB.admissionRate gender
  (admittedA + admittedB) / totalApplicants

/-- Theorem: The weighted average admission rate of Major B is higher than that of Major A -/
theorem majorB_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  weightedAverageAdmissionRate majorB > weightedAverageAdmissionRate majorA := by
  sorry

/-- Theorem: The overall admission rate of males is higher than that of females -/
theorem male_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  overallAdmissionRate majorA majorB Gender.Male > overallAdmissionRate majorA majorB Gender.Female := by
  sorry

end majorB_higher_admission_rate_male_higher_admission_rate_l1835_183503


namespace family_ages_solution_l1835_183524

structure Family where
  father_age : ℕ
  mother_age : ℕ
  john_age : ℕ
  ben_age : ℕ
  mary_age : ℕ

def age_difference (f : Family) : ℕ :=
  f.father_age - f.mother_age

theorem family_ages_solution (f : Family) 
  (h1 : age_difference f = f.john_age - f.ben_age)
  (h2 : age_difference f = f.ben_age - f.mary_age)
  (h3 : f.john_age * f.ben_age = f.father_age)
  (h4 : f.ben_age * f.mary_age = f.mother_age)
  (h5 : f.father_age + f.mother_age + f.john_age + f.ben_age + f.mary_age = 90)
  : f.father_age = 36 ∧ f.mother_age = 36 ∧ f.john_age = 6 ∧ f.ben_age = 6 ∧ f.mary_age = 6 := by
  sorry

end family_ages_solution_l1835_183524


namespace double_earnings_days_theorem_l1835_183551

/-- Calculate the number of additional days needed to double earnings -/
def daysToDoubleEarnings (daysSoFar : ℕ) (earningsSoFar : ℚ) : ℕ :=
  daysSoFar

/-- Theorem: The number of additional days needed to double earnings
    is equal to the number of days already worked -/
theorem double_earnings_days_theorem (daysSoFar : ℕ) (earningsSoFar : ℚ) 
    (hDays : daysSoFar > 0) (hEarnings : earningsSoFar > 0) :
  daysToDoubleEarnings daysSoFar earningsSoFar = daysSoFar := by
  sorry

#eval daysToDoubleEarnings 10 250  -- Should output 10

end double_earnings_days_theorem_l1835_183551


namespace f_2017_neg_two_eq_three_fifths_l1835_183504

def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ f_n n

theorem f_2017_neg_two_eq_three_fifths :
  f_n 2017 (-2) = 3/5 := by sorry

end f_2017_neg_two_eq_three_fifths_l1835_183504


namespace ocean_area_ratio_l1835_183554

theorem ocean_area_ratio (total_area land_area ocean_area : ℝ)
  (land_ratio : land_area / total_area = 29 / 100)
  (ocean_ratio : ocean_area / total_area = 71 / 100)
  (northern_land : ℝ) (southern_land : ℝ)
  (northern_land_ratio : northern_land / land_area = 3 / 4)
  (southern_land_ratio : southern_land / land_area = 1 / 4)
  (northern_ocean southern_ocean : ℝ)
  (northern_hemisphere : northern_land + northern_ocean = total_area / 2)
  (southern_hemisphere : southern_land + southern_ocean = total_area / 2) :
  southern_ocean / northern_ocean = 171 / 113 :=
sorry

end ocean_area_ratio_l1835_183554


namespace subtraction_to_sum_equality_l1835_183590

theorem subtraction_to_sum_equality : 3 - 10 - 7 = 3 + (-10) + (-7) := by
  sorry

end subtraction_to_sum_equality_l1835_183590


namespace fair_distributions_is_square_l1835_183517

/-- The number of permutations of 2n elements with all cycles of even length -/
def fair_distributions (n : ℕ) : ℕ := sorry

/-- The double factorial function -/
def double_factorial (n : ℕ) : ℕ := sorry

theorem fair_distributions_is_square (n : ℕ) : 
  fair_distributions n = (double_factorial (2 * n - 1))^2 := by sorry

end fair_distributions_is_square_l1835_183517


namespace inequality_equivalence_l1835_183536

open Real

theorem inequality_equivalence (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / (exp x) < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (exp 1 - 1) :=
by sorry

end inequality_equivalence_l1835_183536


namespace two_eggs_remain_l1835_183525

/-- The number of eggs remaining unsold when packaging a given number of eggs into cartons of a specific size -/
def remaining_eggs (debra_eggs eli_eggs fiona_eggs carton_size : ℕ) : ℕ :=
  (debra_eggs + eli_eggs + fiona_eggs) % carton_size

/-- Theorem stating that given the specified number of eggs and carton size, 2 eggs will remain unsold -/
theorem two_eggs_remain :
  remaining_eggs 45 58 19 15 = 2 := by
  sorry

end two_eggs_remain_l1835_183525


namespace binomial_coefficient_equality_l1835_183560

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) → n = 7 := by
  sorry

end binomial_coefficient_equality_l1835_183560


namespace parabola_directrix_l1835_183541

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the line containing the focus
def focus_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem parabola_directrix (a : ℝ) :
  (∃ x y, focus_line x y ∧ (x = 0 ∨ y = 0)) →
  (∃ x, ∀ y, y = parabola a x ↔ y + 1 = 2 * (parabola a (x/2))) :=
sorry

end parabola_directrix_l1835_183541


namespace rohits_walk_l1835_183579

/-- Given a right triangle with one leg of length 20 and hypotenuse of length 35,
    the length of the other leg is √825. -/
theorem rohits_walk (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 20) (h3 : c = 35) :
  b = Real.sqrt 825 := by
  sorry

end rohits_walk_l1835_183579


namespace complex_division_equality_l1835_183561

theorem complex_division_equality : (2 : ℂ) / (2 - I) = 4/5 + 2/5 * I := by sorry

end complex_division_equality_l1835_183561


namespace sum_of_percentages_l1835_183537

theorem sum_of_percentages (X Y Z : ℝ) : 
  X = 0.2 * 50 →
  40 = 0.2 * Y →
  40 = (Z / 100) * 50 →
  X + Y + Z = 290 := by
  sorry

end sum_of_percentages_l1835_183537


namespace no_formula_matches_all_points_l1835_183539

-- Define the table of values
def table : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 100), (3, 20), (4, -80)]

-- Define the formulas
def formula_A (x : ℤ) : ℤ := 200 - 30*x
def formula_B (x : ℤ) : ℤ := 200 - 20*x - 10*x^2
def formula_C (x : ℤ) : ℤ := 200 - 40*x + 10*x^2
def formula_D (x : ℤ) : ℤ := 200 - 10*x - 20*x^2

-- Theorem statement
theorem no_formula_matches_all_points :
  ¬(∀ (x y : ℤ), (x, y) ∈ table → 
    (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by sorry

end no_formula_matches_all_points_l1835_183539


namespace grid_last_row_digits_l1835_183511

/-- Represents a 3x4 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 4) ℕ

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 7 \ {0}) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  g 1 1 = 5 ∧
  g 2 3 = 6

theorem grid_last_row_digits (g : Grid) (h : is_valid_grid g) :
  g 2 0 * 10000 + g 2 1 * 1000 + g 2 2 * 100 + g 2 3 * 10 + g 1 3 = 46123 :=
by sorry

end grid_last_row_digits_l1835_183511


namespace wong_valentines_l1835_183510

/-- The number of Valentines Mrs. Wong gave away -/
def valentines_given : ℕ := 8

/-- The number of Valentines Mrs. Wong had left -/
def valentines_left : ℕ := 22

/-- The initial number of Valentines Mrs. Wong had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem wong_valentines : initial_valentines = 30 := by
  sorry

end wong_valentines_l1835_183510


namespace xyz_value_l1835_183572

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) :
  x * y * z = 20 := by
sorry

end xyz_value_l1835_183572


namespace total_books_count_l1835_183569

/-- Given Benny's initial book count, the number of books he gave to Sandy, and Tim's book count,
    prove that the total number of books Benny and Tim have together is 47. -/
theorem total_books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
    (h1 : benny_initial = 24)
    (h2 : given_to_sandy = 10)
    (h3 : tim_books = 33) :
    benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end total_books_count_l1835_183569


namespace distance_to_nearest_town_l1835_183568

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end distance_to_nearest_town_l1835_183568


namespace original_price_calculation_l1835_183523

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.06) * (1 + 0.10) = 6876.1) → P = 6650 := by
  sorry

end original_price_calculation_l1835_183523


namespace imo_2007_problem_5_l1835_183567

theorem imo_2007_problem_5 (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k := by
  sorry

end imo_2007_problem_5_l1835_183567
