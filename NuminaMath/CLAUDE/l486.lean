import Mathlib

namespace identical_pairs_x_value_l486_48690

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := λ (a, b) (c, d) ↦ (a - c, b + d)

theorem identical_pairs_x_value :
  ∀ x y : ℤ, star (2, 2) (4, 1) = star (x, y) (1, 4) → x = -1 :=
by sorry

end identical_pairs_x_value_l486_48690


namespace dividend_from_quotient_and_remainder_l486_48625

theorem dividend_from_quotient_and_remainder :
  ∀ (dividend quotient remainder : ℕ),
    dividend = 23 * quotient + remainder →
    quotient = 38 →
    remainder = 7 →
    dividend = 881 := by
  sorry

end dividend_from_quotient_and_remainder_l486_48625


namespace negate_200_times_minus_one_l486_48623

/-- Represents the result of negating a number n times -/
def negate_n_times (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => λ x => -(negate_n_times n x)

/-- The theorem states that negating -1 200 times results in -1 -/
theorem negate_200_times_minus_one :
  negate_n_times 200 (-1) = -1 := by sorry

end negate_200_times_minus_one_l486_48623


namespace circles_intersect_l486_48658

/-- Two circles in a 2D plane --/
structure TwoCircles where
  /-- First circle: (x-1)^2 + y^2 = 1 --/
  c1 : (ℝ × ℝ) → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 1
  /-- Second circle: x^2 + y^2 + 2x + 4y - 4 = 0 --/
  c2 : (ℝ × ℝ) → Prop := fun p => p.1^2 + p.2^2 + 2*p.1 + 4*p.2 - 4 = 0

/-- The two circles intersect --/
theorem circles_intersect (tc : TwoCircles) : ∃ p : ℝ × ℝ, tc.c1 p ∧ tc.c2 p := by
  sorry

end circles_intersect_l486_48658


namespace no_integer_roots_l486_48689

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 3*x^2 - 16*x + 20 = 0 := by
  sorry

end no_integer_roots_l486_48689


namespace vitya_catches_up_in_5_minutes_l486_48609

-- Define the initial walking speed
def initial_speed : ℝ := 1

-- Define the time they walk before Vitya turns back
def initial_time : ℝ := 10

-- Define Vitya's speed multiplier when he starts chasing
def speed_multiplier : ℝ := 5

-- Define the theorem
theorem vitya_catches_up_in_5_minutes :
  let distance := 2 * initial_speed * initial_time
  let relative_speed := speed_multiplier * initial_speed - initial_speed
  distance / relative_speed = 5 := by
sorry

end vitya_catches_up_in_5_minutes_l486_48609


namespace rain_probability_l486_48643

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end rain_probability_l486_48643


namespace round_310242_to_nearest_thousand_l486_48678

def round_to_nearest_thousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 := by
  sorry

end round_310242_to_nearest_thousand_l486_48678


namespace kenya_has_more_peanuts_l486_48608

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := 133

/-- The difference in peanuts between Kenya and Jose -/
def peanut_difference : ℕ := kenya_peanuts - jose_peanuts

theorem kenya_has_more_peanuts : peanut_difference = 48 := by
  sorry

end kenya_has_more_peanuts_l486_48608


namespace smallest_number_l486_48683

def number_set : Finset ℕ := {5, 9, 10, 2}

theorem smallest_number : 
  ∃ (x : ℕ), x ∈ number_set ∧ ∀ y ∈ number_set, x ≤ y ∧ x = 2 := by
  sorry

end smallest_number_l486_48683


namespace triangle_angle_120_l486_48617

/-- In a triangle ABC with side lengths a, b, and c, if a^2 = b^2 + bc + c^2, then angle A is 120° -/
theorem triangle_angle_120 (a b c : ℝ) (h : a^2 = b^2 + b*c + c^2) :
  let A := Real.arccos ((c^2 + b^2 - a^2) / (2*b*c))
  A = 2*π/3 := by sorry

end triangle_angle_120_l486_48617


namespace cut_cylinder_unpainted_face_area_l486_48614

/-- The area of an unpainted face of a cut cylinder -/
theorem cut_cylinder_unpainted_face_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  let sector_area := π * r^2 / 4
  let triangle_area := r^2 / 2
  let unpainted_face_area := h * (sector_area + triangle_area)
  unpainted_face_area = 62.5 * π + 125 := by
  sorry

end cut_cylinder_unpainted_face_area_l486_48614


namespace purse_cost_multiple_l486_48656

theorem purse_cost_multiple (wallet_cost purse_cost : ℚ) : 
  wallet_cost = 22 →
  wallet_cost + purse_cost = 107 →
  ∃ n : ℕ, n ≤ 4 ∧ purse_cost < n * wallet_cost :=
by sorry

end purse_cost_multiple_l486_48656


namespace product_bounds_l486_48616

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2+Real.sqrt 3)/8 := by
  sorry

end product_bounds_l486_48616


namespace hockey_players_count_l486_48639

theorem hockey_players_count (cricket football softball total : ℕ) 
  (h_cricket : cricket = 16)
  (h_football : football = 18)
  (h_softball : softball = 13)
  (h_total : total = 59)
  : total - (cricket + football + softball) = 12 := by
  sorry

end hockey_players_count_l486_48639


namespace square_with_hole_l486_48668

theorem square_with_hole (n m : ℕ) (h1 : n^2 - m^2 = 209) (h2 : n > m) : n^2 = 225 := by
  sorry

end square_with_hole_l486_48668


namespace spider_plant_babies_l486_48662

/-- The number of baby plants produced by a spider plant in a given time period -/
def baby_plants (plants_per_time : ℕ) (times_per_year : ℕ) (years : ℕ) : ℕ :=
  plants_per_time * times_per_year * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will have 16 baby plants after 4 years -/
theorem spider_plant_babies : baby_plants 2 2 4 = 16 := by
  sorry

end spider_plant_babies_l486_48662


namespace digits_of_large_number_l486_48634

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- 8^22 * 5^19 expressed as a natural number -/
def large_number : ℕ := 8^22 * 5^19

theorem digits_of_large_number :
  num_digits large_number = 35 := by sorry

end digits_of_large_number_l486_48634


namespace hyperbola_equation_l486_48654

/-- A hyperbola with foci on the x-axis, passing through (4√2, -3), and having perpendicular lines
    connecting (0, 5) to its foci, has the standard equation x²/16 - y²/9 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c^2 = a^2 + b^2) : 
  (32 / a^2 - 9 / b^2 = 1) → (25 / c^2 = 1) → (a = 4 ∧ b = 3) := by
  sorry

#check hyperbola_equation

end hyperbola_equation_l486_48654


namespace f_20_l486_48631

/-- A linear function with specific properties -/
def f (x : ℝ) : ℝ := sorry

/-- The function f satisfies f(0) = 3 -/
axiom f_0 : f 0 = 3

/-- The function f increases by 10 when x increases by 4 -/
axiom f_increase (x : ℝ) : f (x + 4) - f x = 10

/-- Theorem: f(20) = 53 -/
theorem f_20 : f 20 = 53 := by sorry

end f_20_l486_48631


namespace max_sum_of_cubes_max_sum_of_cubes_attained_l486_48697

theorem max_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
by sorry

theorem max_sum_of_cubes_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 9 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 > 27 - ε :=
by sorry

end max_sum_of_cubes_max_sum_of_cubes_attained_l486_48697


namespace angle_range_l486_48612

theorem angle_range (θ α : Real) : 
  (∃ (x y : Real), x = Real.sin (α - π/3) ∧ y = Real.sqrt 3 ∧ 
    x = Real.sin θ ∧ y = Real.cos θ) →
  Real.sin (2*θ) ≤ 0 →
  -2*π/3 ≤ α ∧ α ≤ π/3 := by
sorry

end angle_range_l486_48612


namespace sharon_oranges_l486_48660

theorem sharon_oranges (janet_oranges total_oranges : ℕ) 
  (h1 : janet_oranges = 9) 
  (h2 : total_oranges = 16) : 
  total_oranges - janet_oranges = 7 := by
  sorry

end sharon_oranges_l486_48660


namespace jenny_mother_age_problem_l486_48633

/-- Given that Jenny is 10 years old in 2010 and her mother's age is five times Jenny's age,
    prove that the year when Jenny's mother's age will be twice Jenny's age is 2040. -/
theorem jenny_mother_age_problem (jenny_age_2010 : ℕ) (mother_age_2010 : ℕ) :
  jenny_age_2010 = 10 →
  mother_age_2010 = 5 * jenny_age_2010 →
  ∃ (years_after_2010 : ℕ),
    mother_age_2010 + years_after_2010 = 2 * (jenny_age_2010 + years_after_2010) ∧
    2010 + years_after_2010 = 2040 :=
by sorry

end jenny_mother_age_problem_l486_48633


namespace christmas_gifts_theorem_l486_48684

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial_gifts additional_gifts gifts_sent : ℕ) : ℕ :=
  initial_gifts + additional_gifts - gifts_sent

/-- Theorem: Given the initial gifts, additional gifts, and gifts sent,
    prove that the number of gifts left under the tree is 44. -/
theorem christmas_gifts_theorem :
  gifts_left 77 33 66 = 44 := by
  sorry

end christmas_gifts_theorem_l486_48684


namespace candy_cost_theorem_l486_48627

/-- Calculates the cost of purchasing chocolate candies with a bulk discount -/
def calculate_candy_cost (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let boxes_needed := total_candies / candies_per_box
  let total_cost := boxes_needed * cost_per_box
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

/-- The cost of purchasing 450 chocolate candies is $67.5 -/
theorem candy_cost_theorem :
  calculate_candy_cost 30 5 450 300 (1/10) = 67.5 := by
  sorry

end candy_cost_theorem_l486_48627


namespace angies_age_problem_l486_48611

theorem angies_age_problem (angie_age : ℕ) (certain_number : ℕ) : 
  angie_age = 8 → 2 * angie_age + certain_number = 20 → certain_number = 4 := by
  sorry

end angies_age_problem_l486_48611


namespace negation_equivalence_l486_48635

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 ≤ |x|) ↔ (∀ x ∈ Set.Ioo (-1 : ℝ) 0, x^2 > |x|) :=
by sorry

end negation_equivalence_l486_48635


namespace complex_square_problem_l486_48610

theorem complex_square_problem (z : ℂ) (h : z⁻¹ = 1 + Complex.I) : z^2 = -Complex.I / 2 := by
  sorry

end complex_square_problem_l486_48610


namespace tan_x_minus_pi_4_eq_one_third_l486_48686

theorem tan_x_minus_pi_4_eq_one_third (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.cos (2 * x - Real.pi / 2) = Real.sin x ^ 2) : 
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end tan_x_minus_pi_4_eq_one_third_l486_48686


namespace inequality_solution_l486_48694

theorem inequality_solution (x : ℤ) : 
  Real.sqrt (3 * x - 7) - Real.sqrt (3 * x^2 - 13 * x + 13) ≥ 3 * x^2 - 16 * x + 20 → x = 3 := by
  sorry

end inequality_solution_l486_48694


namespace smallest_number_less_than_negative_one_l486_48648

theorem smallest_number_less_than_negative_one :
  let numbers : List ℝ := [-1/2, 0, |(-2)|, -3]
  ∀ x ∈ numbers, x < -1 ↔ x = -3 := by
  sorry

end smallest_number_less_than_negative_one_l486_48648


namespace floor_ceil_sum_l486_48692

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(2.002 : ℝ)⌉ = 3 := by
  sorry

end floor_ceil_sum_l486_48692


namespace car_speed_l486_48642

/-- Given a car traveling 810 km in 5 hours, its speed is 162 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 810) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 162 := by
sorry

end car_speed_l486_48642


namespace mersenne_prime_condition_l486_48695

theorem mersenne_prime_condition (a b : ℕ) (h1 : a ≥ 1) (h2 : b ≥ 2) 
  (h3 : Nat.Prime (a^b - 1)) : a = 2 ∧ Nat.Prime b := by
  sorry

end mersenne_prime_condition_l486_48695


namespace product_price_interval_l486_48652

theorem product_price_interval (price : ℝ) 
  (h1 : price < 2000)
  (h2 : price > 1000)
  (h3 : price < 1500)
  (h4 : price > 1250)
  (h5 : price > 1375) :
  price ∈ Set.Ioo 1375 1500 := by
sorry

end product_price_interval_l486_48652


namespace problem_1_problem_2_problem_3_l486_48630

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Theorem statements
theorem problem_1 (k : ℝ) : 
  (∀ x ∈ I, f k x ≤ g x) ↔ k ≥ 45 :=
sorry

theorem problem_2 (k : ℝ) :
  (∃ x ∈ I, f k x ≤ g x) ↔ k ≥ -7 :=
sorry

theorem problem_3 (k : ℝ) :
  (∀ x1 ∈ I, ∀ x2 ∈ I, f k x1 ≤ g x2) ↔ k ≥ 141 :=
sorry

end problem_1_problem_2_problem_3_l486_48630


namespace freshmen_liberal_arts_percentage_l486_48664

theorem freshmen_liberal_arts_percentage 
  (total_students : ℝ) 
  (freshmen_percentage : ℝ) 
  (liberal_arts_freshmen_percentage : ℝ) 
  (psychology_majors_percentage : ℝ) 
  (freshmen_psychology_liberal_arts_percentage : ℝ) 
  (h1 : freshmen_percentage = 0.5)
  (h2 : psychology_majors_percentage = 0.2)
  (h3 : freshmen_psychology_liberal_arts_percentage = 0.04)
  (h4 : freshmen_psychology_liberal_arts_percentage * total_students = 
        psychology_majors_percentage * liberal_arts_freshmen_percentage * freshmen_percentage * total_students) :
  liberal_arts_freshmen_percentage = 0.4 := by
sorry

end freshmen_liberal_arts_percentage_l486_48664


namespace arithmetic_sequence_sum_ratio_l486_48669

/-- Given an arithmetic sequence {a_n} where a_5/a_3 = 5/9, prove that S_9/S_5 = 1 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (h : a 5 / a 3 = 5 / 9) :
  let S : ℕ → ℝ := λ n => (n / 2) * (a 1 + a n)
  S 9 / S 5 = 1 := by sorry

end arithmetic_sequence_sum_ratio_l486_48669


namespace parabola_hyperbola_focus_l486_48651

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) (n : ℝ) : Prop := x^2/3 - y^2/n = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define a predicate for a point being a focus of the hyperbola
def is_hyperbola_focus (x y : ℝ) (n : ℝ) : Prop :=
  hyperbola x y n ∧ x^2 - y^2 = 3 + n

-- State the theorem
theorem parabola_hyperbola_focus (n : ℝ) :
  (∃ x y, is_hyperbola_focus x y n ∧ (x, y) = parabola_focus) → n = 1 :=
sorry

end parabola_hyperbola_focus_l486_48651


namespace real_part_of_z_l486_48649

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.re = 5 / 2 := by
sorry

end real_part_of_z_l486_48649


namespace product_calculation_l486_48657

theorem product_calculation : 3.5 * 7.2 * (6.3 - 1.4) = 122.5 := by
  sorry

end product_calculation_l486_48657


namespace r_power_sum_l486_48646

theorem r_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_power_sum_l486_48646


namespace a_perpendicular_b_l486_48696

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Vector a in ℝ² -/
def a : ℝ × ℝ := (3, 4)

/-- Vector b in ℝ² -/
def b : ℝ × ℝ := (-8, 6)

/-- Theorem: Vectors a and b are perpendicular -/
theorem a_perpendicular_b : perpendicular a b := by
  sorry

end a_perpendicular_b_l486_48696


namespace grape_juice_mixture_l486_48655

theorem grape_juice_mixture (initial_volume : ℝ) (added_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 40 →
  added_volume = 10 →
  final_percentage = 0.36 →
  let final_volume := initial_volume + added_volume
  let initial_percentage := (final_percentage * final_volume - added_volume) / initial_volume
  initial_percentage = 0.2 := by
sorry

end grape_juice_mixture_l486_48655


namespace volleyball_team_selection_l486_48622

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the volleyball club -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- The number of non-quadruplet players -/
def other_players : ℕ := total_players - num_quadruplets

theorem volleyball_team_selection :
  (binomial total_players num_starters) -
  (binomial other_players (num_starters - num_quadruplets)) -
  (binomial other_players num_starters) = 15470 := by sorry

end volleyball_team_selection_l486_48622


namespace farm_animal_difference_l486_48645

/-- Represents the number of horses and cows on a farm before and after a transaction --/
structure FarmAnimals where
  initial_horses : ℕ
  initial_cows : ℕ
  final_horses : ℕ
  final_cows : ℕ

/-- The conditions of the farm animal problem --/
def farm_conditions (farm : FarmAnimals) : Prop :=
  farm.initial_horses = 6 * farm.initial_cows ∧
  farm.final_horses = farm.initial_horses - 15 ∧
  farm.final_cows = farm.initial_cows + 15 ∧
  farm.final_horses = 3 * farm.final_cows

theorem farm_animal_difference (farm : FarmAnimals) 
  (h : farm_conditions farm) : farm.final_horses - farm.final_cows = 70 := by
  sorry

end farm_animal_difference_l486_48645


namespace cos_inequality_range_l486_48673

theorem cos_inequality_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) := by
sorry

end cos_inequality_range_l486_48673


namespace ellipse_left_vertex_l486_48693

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The theorem stating the properties of the ellipse and its left vertex -/
theorem ellipse_left_vertex 
  (E : Ellipse) 
  (C : Circle) 
  (h_focus : C.h = 3 ∧ C.k = 0) -- One focus is the center of the circle
  (h_circle_eq : ∀ x y, x^2 + y^2 - 6*x + 8 = 0 ↔ (x - C.h)^2 + (y - C.k)^2 = C.r^2) -- Circle equation
  (h_minor_axis : E.b = 4) -- Minor axis is 8 in length
  : ∃ x, x = -5 ∧ (x / E.a)^2 + 0^2 / E.b^2 = 1 -- Left vertex is at (-5, 0)
:= by sorry

end ellipse_left_vertex_l486_48693


namespace total_cantaloupes_l486_48638

def keith_cantaloupes : ℝ := 29.5
def fred_cantaloupes : ℝ := 16.25
def jason_cantaloupes : ℝ := 20.75
def olivia_cantaloupes : ℝ := 12.5
def emily_cantaloupes : ℝ := 15.8

theorem total_cantaloupes : 
  keith_cantaloupes + fred_cantaloupes + jason_cantaloupes + olivia_cantaloupes + emily_cantaloupes = 94.8 := by
  sorry

end total_cantaloupes_l486_48638


namespace cube_root_over_fifth_root_of_five_l486_48659

theorem cube_root_over_fifth_root_of_five (x : ℝ) (hx : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end cube_root_over_fifth_root_of_five_l486_48659


namespace max_side_length_of_triangle_l486_48699

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧  -- Three different integer side lengths
  a + b + c = 24 ∧ -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  c ≤ 10 :=        -- Maximum length of any side is 10
by sorry

end max_side_length_of_triangle_l486_48699


namespace grassy_width_is_60_l486_48667

/-- Represents a rectangular plot with a gravel path around it. -/
structure RectangularPlot where
  length : ℝ
  totalWidth : ℝ
  pathWidth : ℝ

/-- Calculates the width of the grassy area in a rectangular plot. -/
def grassyWidth (plot : RectangularPlot) : ℝ :=
  plot.totalWidth - 2 * plot.pathWidth

/-- Theorem stating that for a given rectangular plot with specified dimensions,
    the width of the grassy area is 60 meters. -/
theorem grassy_width_is_60 (plot : RectangularPlot)
    (h1 : plot.length = 110)
    (h2 : plot.totalWidth = 65)
    (h3 : plot.pathWidth = 2.5) :
  grassyWidth plot = 60 := by
  sorry

end grassy_width_is_60_l486_48667


namespace complex_modulus_problem_l486_48605

theorem complex_modulus_problem (z : ℂ) : (z - 3) * (1 - 3*I) = 10 → Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l486_48605


namespace product_a4b4_equals_negative_six_l486_48661

theorem product_a4b4_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end product_a4b4_equals_negative_six_l486_48661


namespace special_triangle_existence_l486_48644

/-- A triangle with integer side lengths satisfying a special condition -/
def SpecialTriangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive integers
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  a * b * c = 2 * (a - 1) * (b - 1) * (c - 1)  -- Special condition

theorem special_triangle_existence :
  ∃ a b c : ℕ, SpecialTriangle a b c ∧
  (∀ x y z : ℕ, SpecialTriangle x y z → (x, y, z) = (8, 7, 3) ∨ (x, y, z) = (6, 5, 4)) :=
by sorry


end special_triangle_existence_l486_48644


namespace boat_problem_l486_48641

theorem boat_problem (total_students : ℕ) (big_boat_capacity small_boat_capacity : ℕ) (total_boats : ℕ) :
  total_students = 52 →
  big_boat_capacity = 8 →
  small_boat_capacity = 4 →
  total_boats = 9 →
  ∃ (big_boats small_boats : ℕ),
    big_boats + small_boats = total_boats ∧
    big_boats * big_boat_capacity + small_boats * small_boat_capacity = total_students ∧
    big_boats = 4 :=
by sorry

end boat_problem_l486_48641


namespace limit_Sn_divided_by_n2Bn_l486_48681

-- Define the set A
def A (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define B_n as the number of subsets of A
def B_n (n : ℕ) : ℕ := 2^n

-- Define S_n as the sum of elements in non-empty proper subsets of A
def S_n (n : ℕ) : ℕ := (n * (n + 1) / 2) * (2^(n - 1) - 1)

-- State the theorem
theorem limit_Sn_divided_by_n2Bn (ε : ℝ) (ε_pos : ε > 0) :
  ∃ N : ℕ, ∀ n ≥ N, |S_n n / (n^2 * B_n n : ℝ) - 1/4| < ε :=
sorry

end limit_Sn_divided_by_n2Bn_l486_48681


namespace bankers_discount_example_l486_48688

/-- Given a bill with face value and true discount, calculates the banker's discount -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  true_discount + (true_discount^2 / present_value)

/-- Theorem stating that for a bill with face value 540 and true discount 90, 
    the banker's discount is 108 -/
theorem bankers_discount_example : bankers_discount 540 90 = 108 := by
  sorry

end bankers_discount_example_l486_48688


namespace fourth_shot_probability_l486_48624

/-- The probability of making a shot given the previous shot was made -/
def p_make_given_make : ℚ := 2/3

/-- The probability of making a shot given the previous shot was missed -/
def p_make_given_miss : ℚ := 1/3

/-- The probability of making the first shot -/
def p_first_shot : ℚ := 2/3

/-- The probability of making the n-th shot -/
def p_nth_shot (n : ℕ) : ℚ :=
  1/2 * (1 + 1 / 3^n)

theorem fourth_shot_probability :
  p_nth_shot 4 = 41/81 :=
sorry

end fourth_shot_probability_l486_48624


namespace complex_number_problem_l486_48640

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2 * Complex.I = r)
  (h2 : ∃ (m : ℝ), z / (2 - Complex.I) = m) : 
  z = 4 - 2 * Complex.I := by
sorry

end complex_number_problem_l486_48640


namespace least_value_theorem_l486_48604

theorem least_value_theorem (x y z : ℕ+) 
  (h1 : 5 * y.val = 6 * z.val)
  (h2 : x.val + y.val + z.val = 26) :
  5 * y.val = 30 := by
sorry

end least_value_theorem_l486_48604


namespace construct_75_degree_angle_l486_48675

/-- Given an angle of 19°, it is possible to construct an angle of 75°. -/
theorem construct_75_degree_angle (angle : ℝ) (h : angle = 19) : 
  ∃ (constructed_angle : ℝ), constructed_angle = 75 := by
sorry

end construct_75_degree_angle_l486_48675


namespace sequence_general_term_l486_48670

theorem sequence_general_term (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ k, S k = 2 * k^2 - 3 * k) : 
  a n = 4 * n - 5 := by
  sorry

end sequence_general_term_l486_48670


namespace log_sum_equals_two_l486_48618

theorem log_sum_equals_two : 2 * Real.log 63 + Real.log 64 = 2 := by
  sorry

end log_sum_equals_two_l486_48618


namespace at_least_eight_nonzero_digits_l486_48680

/-- Given a natural number n, returns a number consisting of n repeating 9's -/
def repeating_nines (n : ℕ) : ℕ := 10^n - 1

/-- Counts the number of non-zero digits in the decimal representation of a natural number -/
def count_nonzero_digits (k : ℕ) : ℕ := sorry

theorem at_least_eight_nonzero_digits 
  (k : ℕ) (n : ℕ) (h1 : k > 0) (h2 : k % repeating_nines n = 0) : 
  count_nonzero_digits k ≥ 8 := by sorry

end at_least_eight_nonzero_digits_l486_48680


namespace roses_count_l486_48698

/-- The number of pots of roses in the People's Park -/
def roses : ℕ := 65

/-- The number of pots of lilac flowers in the People's Park -/
def lilacs : ℕ := 180

/-- Theorem stating that the number of pots of roses is correct given the conditions -/
theorem roses_count :
  roses = 65 ∧ lilacs = 180 ∧ lilacs = 3 * roses - 15 :=
by sorry

end roses_count_l486_48698


namespace factorial_prime_factorization_l486_48665

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_prime_factorization :
  ∃ (i k m p q : ℕ+),
    factorial 12 = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) * 11^(q.val) ∧
    i.val + k.val + m.val + p.val + q.val = 28 := by
  sorry

end factorial_prime_factorization_l486_48665


namespace count_satisfying_numbers_l486_48613

/-- Represents a two-digit number in the dozenal (base 12) system -/
structure DozenalNumber :=
  (tens : Nat)
  (ones : Nat)
  (tens_valid : 1 ≤ tens ∧ tens ≤ 11)
  (ones_valid : ones ≤ 11)

/-- Converts a DozenalNumber to its decimal representation -/
def toDecimal (n : DozenalNumber) : Nat :=
  12 * n.tens + n.ones

/-- Calculates the sum of digits of a DozenalNumber -/
def digitSum (n : DozenalNumber) : Nat :=
  n.tens + n.ones

/-- Checks if a DozenalNumber satisfies the given condition -/
def satisfiesCondition (n : DozenalNumber) : Prop :=
  (toDecimal n - digitSum n) % 12 = 5

theorem count_satisfying_numbers :
  ∃ (numbers : Finset DozenalNumber),
    numbers.card = 12 ∧
    (∀ n : DozenalNumber, n ∈ numbers ↔ satisfiesCondition n) :=
sorry

end count_satisfying_numbers_l486_48613


namespace circle_diameter_l486_48626

theorem circle_diameter (A : ℝ) (r : ℝ) (D : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → D = 2 * r → D = 20 := by
  sorry

end circle_diameter_l486_48626


namespace lapis_share_is_correct_l486_48637

/-- Represents the share of treasure for a person -/
structure TreasureShare where
  amount : ℚ
  deriving Repr

/-- Calculates the share of treasure based on contribution -/
def calculateShare (contribution : ℚ) (totalContribution : ℚ) (treasureValue : ℚ) : TreasureShare :=
  { amount := (contribution / totalContribution) * treasureValue }

theorem lapis_share_is_correct (fonzie_contribution : ℚ) (aunt_bee_contribution : ℚ) (lapis_contribution : ℚ) (treasure_value : ℚ)
    (h1 : fonzie_contribution = 7000)
    (h2 : aunt_bee_contribution = 8000)
    (h3 : lapis_contribution = 9000)
    (h4 : treasure_value = 900000) :
  (calculateShare lapis_contribution (fonzie_contribution + aunt_bee_contribution + lapis_contribution) treasure_value).amount = 337500 := by
  sorry

#eval calculateShare 9000 24000 900000

end lapis_share_is_correct_l486_48637


namespace max_sundays_in_53_days_l486_48636

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def total_days : ℕ := 53

/-- A function that returns the number of Sundays in a given number of days -/
def sundays_in_days (days : ℕ) : ℕ := days / days_in_week

theorem max_sundays_in_53_days : 
  sundays_in_days total_days = 7 := by sorry

end max_sundays_in_53_days_l486_48636


namespace iris_spending_l486_48672

/-- Calculates the total amount spent by Iris on clothes, including discount and tax --/
def total_spent (jacket_price : ℚ) (jacket_quantity : ℕ)
                (shorts_price : ℚ) (shorts_quantity : ℕ)
                (pants_price : ℚ) (pants_quantity : ℕ)
                (tops_price : ℚ) (tops_quantity : ℕ)
                (skirts_price : ℚ) (skirts_quantity : ℕ)
                (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The theorem stating that Iris spent $230.16 on clothes --/
theorem iris_spending : 
  total_spent 15 3 10 2 18 4 7 6 12 5 (10/100) (7/100) = 230.16 := by
  sorry

end iris_spending_l486_48672


namespace min_tangent_length_is_4_l486_48615

/-- The circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- The line of symmetry for circle C -/
def symmetry_line (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- The point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- The minimum tangent length from a point to a circle -/
def min_tangent_length (p : Point) (C : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_line a b a b →
  min_tangent_length (Point.mk a b) circle_C = 4 := by
  sorry

end min_tangent_length_is_4_l486_48615


namespace tangent_slope_condition_l486_48653

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

-- State the theorem
theorem tangent_slope_condition (a b : ℝ) :
  (∀ x, (deriv (f a b)) x = 2 * a * x) →  -- Derivative of f
  (deriv (f a b)) 1 = 2 →  -- Slope of tangent line at x = 1 is 2
  f a b 1 = 3 →  -- Function value at x = 1 is 3
  b / a = 2 := by
sorry

end tangent_slope_condition_l486_48653


namespace domain_of_z_l486_48677

def z (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x + 1) ^ (1/2)

theorem domain_of_z : 
  {x : ℝ | ∃ y, z x = y} = {x : ℝ | x ≥ 5} :=
sorry

end domain_of_z_l486_48677


namespace inequality_solution_l486_48687

theorem inequality_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x - 1) / 2 + 1 ≥ 2 * x := by
  sorry

end inequality_solution_l486_48687


namespace arithmetic_geometric_mean_inequality_l486_48679

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end arithmetic_geometric_mean_inequality_l486_48679


namespace arithmetic_sequence_problem_l486_48666

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a₃ + a₅ = 10, prove that a₄ = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 10) : 
  a 4 = 5 := by
  sorry

end arithmetic_sequence_problem_l486_48666


namespace max_daily_sales_revenue_l486_48691

-- Define the domain of t
def T : Set ℕ := {t : ℕ | 1 ≤ t ∧ t ≤ 20}

-- Define the daily sales volume function
def f (t : ℕ) : ℝ := -t + 30

-- Define the daily sales price function
def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

-- Define the daily sales revenue function
def S (t : ℕ) : ℝ := f t * g t

-- Theorem stating the maximum daily sales revenue
theorem max_daily_sales_revenue :
  ∃ (t_max : ℕ), t_max ∈ T ∧
    (∀ (t : ℕ), t ∈ T → S t ≤ S t_max) ∧
    t_max = 5 ∧ S t_max = 1250 := by
  sorry

end max_daily_sales_revenue_l486_48691


namespace ceiling_sum_sqrt_l486_48632

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l486_48632


namespace clock_hands_right_angle_count_l486_48676

/-- The number of times clock hands form a right angle in a 12-hour period -/
def right_angles_12h : ℕ := 22

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days we're considering -/
def days : ℕ := 2

theorem clock_hands_right_angle_count :
  (right_angles_12h * hours_per_day * days) / 12 = 88 := by
  sorry

end clock_hands_right_angle_count_l486_48676


namespace triangle_tan_C_l486_48601

/-- Given a triangle ABC with sides a, b, and c satisfying 3a² + 3b² - 3c² + 2ab = 0,
    prove that tan C = -2√2 -/
theorem triangle_tan_C (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 + 2 * a * b = 0) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  Real.tan C = -2 * Real.sqrt 2 := by
sorry

end triangle_tan_C_l486_48601


namespace largest_value_when_x_is_quarter_l486_48603

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) :=
by sorry

end largest_value_when_x_is_quarter_l486_48603


namespace marathon_yards_remainder_l486_48663

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def marathon : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

/-- Calculates the total distance for a given number of marathons -/
def total_distance (n : ℕ) (d : MarathonDistance) : TotalDistance :=
  { miles := n * d.miles,
    yards := n * d.yards }

/-- Converts excess yards to miles and updates the TotalDistance -/
def normalize_distance (d : TotalDistance) : TotalDistance :=
  { miles := d.miles + d.yards / yards_per_mile,
    yards := d.yards % yards_per_mile }

theorem marathon_yards_remainder :
  (normalize_distance (total_distance num_marathons marathon)).yards = 495 := by
  sorry

end marathon_yards_remainder_l486_48663


namespace hat_markup_price_l486_48600

theorem hat_markup_price (P : ℝ) 
  (h1 : 2 * P - (P + 0.7 * P) = 6) : 
  P + 0.7 * P = 34 := by
  sorry

end hat_markup_price_l486_48600


namespace fractional_equation_solution_range_l486_48619

/-- Given a fractional equation and conditions on its solution, 
    this theorem proves the range of values for the parameter a. -/
theorem fractional_equation_solution_range (a x : ℝ) : 
  (a / (x + 2) = 1 - 3 / (x + 2)) →
  (x < 0) →
  (a < -1 ∧ a ≠ -3) := by
  sorry

end fractional_equation_solution_range_l486_48619


namespace max_bananas_is_7_l486_48682

def budget : ℕ := 10
def single_banana_cost : ℕ := 2
def bundle_4_cost : ℕ := 6
def bundle_6_cost : ℕ := 8

def max_bananas (b s b4 b6 : ℕ) : ℕ := 
  sorry

theorem max_bananas_is_7 : 
  max_bananas budget single_banana_cost bundle_4_cost bundle_6_cost = 7 := by
  sorry

end max_bananas_is_7_l486_48682


namespace fourth_term_of_geometric_sequence_l486_48620

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem fourth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) : 
  a 4 = 9 :=
sorry

end fourth_term_of_geometric_sequence_l486_48620


namespace profit_percentage_previous_year_l486_48647

/-- Prove that the profit percentage in the previous year was 10% -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenue in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : 0.8 * R = revenue_2009) -- Revenue in 2009 was 80% of previous year
  (h3 : 0.13 * revenue_2009 = profit_2009) -- Profit in 2009 was 13% of 2009 revenue
  (h4 : profit_2009 = 1.04 * profit_previous) -- Profit in 2009 was 104% of previous year's profit
  (h5 : profit_previous = P / 100 * R) -- Definition of profit percentage
  : P = 10 := by sorry

end profit_percentage_previous_year_l486_48647


namespace lucky_larry_coincidence_l486_48607

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 1 → b = 2 → c = 3 → d = 4 → 
  (a - b - c - d + e = a - (b - (c - (d + e)))) → e = 3 := by
sorry

end lucky_larry_coincidence_l486_48607


namespace coffee_mix_theorem_l486_48650

/-- Calculates the price per pound of a coffee mix given the prices and quantities of two types of coffee. -/
def coffee_mix_price (price1 price2 : ℚ) (quantity1 quantity2 : ℚ) : ℚ :=
  (price1 * quantity1 + price2 * quantity2) / (quantity1 + quantity2)

/-- Theorem stating that mixing equal quantities of two types of coffee priced at $2.15 and $2.45 per pound
    results in a mix priced at $2.30 per pound. -/
theorem coffee_mix_theorem :
  let price1 : ℚ := 215 / 100
  let price2 : ℚ := 245 / 100
  let quantity1 : ℚ := 9
  let quantity2 : ℚ := 9
  coffee_mix_price price1 price2 quantity1 quantity2 = 230 / 100 := by
  sorry

#eval coffee_mix_price (215/100) (245/100) 9 9

end coffee_mix_theorem_l486_48650


namespace candy_mixture_cost_l486_48671

theorem candy_mixture_cost (first_candy_weight : ℝ) (second_candy_weight : ℝ) 
  (second_candy_price : ℝ) (mixture_price : ℝ) :
  first_candy_weight = 20 →
  second_candy_weight = 80 →
  second_candy_price = 5 →
  mixture_price = 6 →
  first_candy_weight + second_candy_weight = 100 →
  ∃ (first_candy_price : ℝ),
    first_candy_price * first_candy_weight + 
    second_candy_price * second_candy_weight = 
    mixture_price * (first_candy_weight + second_candy_weight) ∧
    first_candy_price = 10 := by
  sorry


end candy_mixture_cost_l486_48671


namespace sean_houses_bought_l486_48685

theorem sean_houses_bought (initial_houses : ℕ) (traded_houses : ℕ) (final_houses : ℕ) 
  (h1 : initial_houses = 27)
  (h2 : traded_houses = 8)
  (h3 : final_houses = 31) :
  final_houses - (initial_houses - traded_houses) = 12 :=
by
  sorry

end sean_houses_bought_l486_48685


namespace probability_even_greater_than_10_l486_48628

def ball_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_product (a b : ℕ) : Bool :=
  Even (a * b) ∧ a * b > 10

def valid_outcomes : Finset (ℕ × ℕ) :=
  ball_set.product ball_set

def favorable_outcomes : Finset (ℕ × ℕ) :=
  valid_outcomes.filter (fun p => is_valid_product p.1 p.2)

theorem probability_even_greater_than_10 :
  (favorable_outcomes.card : ℚ) / valid_outcomes.card = 1 / 5 :=
sorry

end probability_even_greater_than_10_l486_48628


namespace height_of_A_l486_48674

/-- The heights of four people A, B, C, and D satisfying certain conditions -/
structure Heights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_equality : A + B = C + D ∨ A + C = B + D ∨ A + D = B + C
  average_difference : (A + B) / 2 = (A + C) / 2 + 4
  D_taller : D = A + 10
  B_C_sum : B + C = 288

/-- The height of A is 139 cm -/
theorem height_of_A (h : Heights) : h.A = 139 := by
  sorry

end height_of_A_l486_48674


namespace multiplication_equality_l486_48606

theorem multiplication_equality (x : ℝ) : x * 240 = 173 * 240 ↔ x = 173 := by
  sorry

end multiplication_equality_l486_48606


namespace equal_numbers_product_l486_48621

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 24 →
  a = 20 →
  b = 25 →
  c = 33 →
  d = e →
  d * e = 441 := by
sorry

end equal_numbers_product_l486_48621


namespace probability_D_given_E_l486_48629

-- Define the regions D and E
def region_D (x y : ℝ) : Prop := y ≤ 1 ∧ y ≥ x^2
def region_E (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the areas of regions D and E
noncomputable def area_D : ℝ := 4/3
noncomputable def area_E : ℝ := 2

-- State the theorem
theorem probability_D_given_E : 
  (area_D / area_E) = 2/3 :=
sorry

end probability_D_given_E_l486_48629


namespace pages_read_on_thursday_l486_48602

theorem pages_read_on_thursday (wednesday_pages friday_pages total_pages : ℕ) 
  (h1 : wednesday_pages = 18)
  (h2 : friday_pages = 23)
  (h3 : total_pages = 60) :
  total_pages - (wednesday_pages + friday_pages) = 19 := by
sorry

end pages_read_on_thursday_l486_48602
