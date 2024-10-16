import Mathlib

namespace NUMINAMATH_CALUDE_cubic_function_properties_l3540_354038

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line parallel to 3x + y + 2 = 0 at x = 1
  (a = -1 ∧ b = 0) ∧  -- Values of a and b
  (∃ x₁ x₂ : ℝ, f a b c x₁ - f a b c x₂ = 4)  -- Difference between max and min is 4
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3540_354038


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3540_354012

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3)^2 = 4 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3540_354012


namespace NUMINAMATH_CALUDE_total_points_is_201_l3540_354033

/- Define the scoring for Mark's team -/
def marks_team_two_pointers : ℕ := 25
def marks_team_three_pointers : ℕ := 8
def marks_team_free_throws : ℕ := 10

/- Define the scoring for the opponents relative to Mark's team -/
def opponents_two_pointers : ℕ := 2 * marks_team_two_pointers
def opponents_three_pointers : ℕ := marks_team_three_pointers / 2
def opponents_free_throws : ℕ := marks_team_free_throws / 2

/- Calculate the total points for both teams -/
def total_points : ℕ := 
  (marks_team_two_pointers * 2 + marks_team_three_pointers * 3 + marks_team_free_throws) +
  (opponents_two_pointers * 2 + opponents_three_pointers * 3 + opponents_free_throws)

/- Theorem stating that the total points scored by both teams is 201 -/
theorem total_points_is_201 : total_points = 201 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_201_l3540_354033


namespace NUMINAMATH_CALUDE_number_difference_l3540_354083

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 17402)
  (b_div_10 : 10 ∣ b)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 14238 := by sorry

end NUMINAMATH_CALUDE_number_difference_l3540_354083


namespace NUMINAMATH_CALUDE_riverdale_school_theorem_l3540_354031

def riverdale_school (total students_in_band students_in_chorus students_in_band_or_chorus : ℕ) : Prop :=
  students_in_band + students_in_chorus - students_in_band_or_chorus = 30

theorem riverdale_school_theorem :
  riverdale_school 250 90 120 180 := by
  sorry

end NUMINAMATH_CALUDE_riverdale_school_theorem_l3540_354031


namespace NUMINAMATH_CALUDE_f_five_values_l3540_354078

def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y^2) = f (x^2 - y) + 4 * (f x) * y^2

theorem f_five_values (f : ℝ → ℝ) (h : FunctionProperty f) : 
  f 5 = 0 ∨ f 5 = 25 := by sorry

end NUMINAMATH_CALUDE_f_five_values_l3540_354078


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l3540_354008

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 15)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 10 := by
  sorry

#check used_car_seller_problem

end NUMINAMATH_CALUDE_used_car_seller_problem_l3540_354008


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3540_354073

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + 
  (1 / (8 * 9 : ℚ)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3540_354073


namespace NUMINAMATH_CALUDE_congruence_unique_solution_l3540_354056

theorem congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1212 [ZMOD 10] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_unique_solution_l3540_354056


namespace NUMINAMATH_CALUDE_recliner_price_drop_l3540_354029

/-- Proves that a 80% increase in sales and a 44% increase in gross revenue
    results in a 20% price drop -/
theorem recliner_price_drop (P N : ℝ) (P' N' : ℝ) :
  N' = 1.8 * N →
  P' * N' = 1.44 * (P * N) →
  P' = 0.8 * P :=
by sorry

end NUMINAMATH_CALUDE_recliner_price_drop_l3540_354029


namespace NUMINAMATH_CALUDE_first_piece_cost_l3540_354060

/-- Given the total spent on clothing, the number of pieces, and the prices of some pieces,
    prove the cost of the first piece. -/
theorem first_piece_cost (total : ℕ) (num_pieces : ℕ) (price_one : ℕ) (price_others : ℕ) :
  total = 610 →
  num_pieces = 7 →
  price_one = 81 →
  price_others = 96 →
  ∃ (first_piece : ℕ), first_piece + price_one + (num_pieces - 2) * price_others = total ∧ first_piece = 49 := by
  sorry

end NUMINAMATH_CALUDE_first_piece_cost_l3540_354060


namespace NUMINAMATH_CALUDE_prime_relative_frequency_l3540_354075

/-- The number of natural numbers considered -/
def total_numbers : ℕ := 4000

/-- The number of prime numbers among the first 4000 natural numbers -/
def prime_count : ℕ := 551

/-- The relative frequency of prime numbers among the first 4000 natural numbers -/
def relative_frequency : ℚ := prime_count / total_numbers

theorem prime_relative_frequency :
  relative_frequency = 551 / 4000 :=
by sorry

end NUMINAMATH_CALUDE_prime_relative_frequency_l3540_354075


namespace NUMINAMATH_CALUDE_point_ratio_on_line_l3540_354019

/-- Given four points P, Q, R, and S on a line in that order, with specific distances between them,
    prove that the ratio of PR to QS is 7/17. -/
theorem point_ratio_on_line (P Q R S : ℝ) : 
  Q - P = 3 →
  R - Q = 4 →
  S - P = 20 →
  P < Q ∧ Q < R ∧ R < S →
  (R - P) / (S - Q) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_point_ratio_on_line_l3540_354019


namespace NUMINAMATH_CALUDE_jan_height_is_42_l3540_354086

def cary_height : ℕ := 72

def bill_height : ℕ := cary_height / 2

def jan_height : ℕ := bill_height + 6

theorem jan_height_is_42 : jan_height = 42 := by
  sorry

end NUMINAMATH_CALUDE_jan_height_is_42_l3540_354086


namespace NUMINAMATH_CALUDE_company_employee_count_l3540_354061

theorem company_employee_count (december_count : ℕ) (percent_increase : ℚ) : 
  december_count = 480 → percent_increase = 15 / 100 → 
  ∃ (january_count : ℕ), 
    (↑december_count : ℚ) = (1 + percent_increase) * ↑january_count ∧ 
    january_count = 417 := by
  sorry

end NUMINAMATH_CALUDE_company_employee_count_l3540_354061


namespace NUMINAMATH_CALUDE_f_plus_three_odd_l3540_354097

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_three_odd 
  (h1 : IsOdd (fun x ↦ f (x + 1)))
  (h2 : IsOdd (fun x ↦ f (x - 1))) :
  IsOdd (fun x ↦ f (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_three_odd_l3540_354097


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l3540_354032

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for the union of complements of A and B
theorem union_complement_A_B :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l3540_354032


namespace NUMINAMATH_CALUDE_pure_imaginary_magnitude_l3540_354059

theorem pure_imaginary_magnitude (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 12 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_magnitude_l3540_354059


namespace NUMINAMATH_CALUDE_class_average_theorem_l3540_354044

theorem class_average_theorem (total_students : ℝ) (h_total : total_students > 0) :
  let group1_percent : ℝ := 25
  let group1_average : ℝ := 80
  let group2_percent : ℝ := 50
  let group2_average : ℝ := 65
  let group3_percent : ℝ := 100 - group1_percent - group2_percent
  let group3_average : ℝ := 90
  let overall_average : ℝ := (group1_percent * group1_average + group2_percent * group2_average + group3_percent * group3_average) / 100
  overall_average = 75 := by
  sorry


end NUMINAMATH_CALUDE_class_average_theorem_l3540_354044


namespace NUMINAMATH_CALUDE_jogger_speed_l3540_354045

/-- Jogger's speed calculation -/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ) :
  let relative_speed : ℝ := (train_length + initial_distance) / passing_time
  let train_speed_mps : ℝ := train_speed * (5/18)
  let jogger_speed_mps : ℝ := train_speed_mps - relative_speed
  let jogger_speed_kmh : ℝ := jogger_speed_mps * (18/5)
  train_length = 120 →
  initial_distance = 280 →
  train_speed = 45 →
  passing_time = 40 →
  jogger_speed_kmh = 9 := by
  sorry

end NUMINAMATH_CALUDE_jogger_speed_l3540_354045


namespace NUMINAMATH_CALUDE_file_space_calculation_l3540_354072

/-- 
Given a total disk space and the space left after backup, 
calculate the space taken by files.
-/
theorem file_space_calculation (total_space : ℕ) (space_left : ℕ) 
  (h1 : total_space = 28) (h2 : space_left = 2) : 
  total_space - space_left = 26 := by
  sorry

#check file_space_calculation

end NUMINAMATH_CALUDE_file_space_calculation_l3540_354072


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3540_354052

theorem last_two_digits_sum (n : ℕ) : (9^25 + 13^25) % 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3540_354052


namespace NUMINAMATH_CALUDE_function_f_properties_l3540_354024

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧
  (∀ x, x > 0 → f x > 0)

theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  (f 0 = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (f 1 = 2 → ∃ a, f (2 - a) = 6 ∧ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_function_f_properties_l3540_354024


namespace NUMINAMATH_CALUDE_two_year_increase_l3540_354096

def yearly_increase (amount : ℚ) : ℚ := amount * (1 + 1/8)

theorem two_year_increase (P : ℚ) (h : P = 2880) : 
  yearly_increase (yearly_increase P) = 3645 := by
  sorry

end NUMINAMATH_CALUDE_two_year_increase_l3540_354096


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l3540_354003

theorem greatest_common_multiple_under_150 (n : ℕ) :
  (n % 10 = 0 ∧ n % 15 = 0 ∧ n < 150) →
  n ≤ 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l3540_354003


namespace NUMINAMATH_CALUDE_expression_evaluation_l3540_354089

theorem expression_evaluation : 
  (0 : ℝ) - 2 - 2 * Real.sin (45 * π / 180) + (π - 3.14) * 0 + (-1)^3 = -3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3540_354089


namespace NUMINAMATH_CALUDE_division_equations_for_26_l3540_354025

theorem division_equations_for_26 : 
  {(x, y) : ℕ × ℕ | 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 26 = x * y + 2} = 
  {(3, 8), (4, 6), (6, 4), (8, 3)} := by sorry

end NUMINAMATH_CALUDE_division_equations_for_26_l3540_354025


namespace NUMINAMATH_CALUDE_second_number_proof_l3540_354046

theorem second_number_proof (a : ℝ) (h : a = 800) :
  ∃! b : ℝ, 0.4 * a = 0.2 * b + 190 :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l3540_354046


namespace NUMINAMATH_CALUDE_flower_pollination_l3540_354007

/-- Represents the types of flowers -/
inductive FlowerType
| Rose
| Sunflower
| Tulip
| Daisy
| Orchid

/-- Represents a bee -/
structure Bee where
  roses_per_hour : ℕ
  sunflowers_per_hour : ℕ
  tulips_per_hour : ℕ
  daisies_per_hour : ℕ
  orchids_per_hour : ℕ

/-- The problem setup -/
def flower_problem : Prop :=
  let total_flowers : ℕ := 60
  let roses : ℕ := 12
  let sunflowers : ℕ := 15
  let tulips : ℕ := 9
  let daisies : ℕ := 18
  let orchids : ℕ := 6
  let hours : ℕ := 3
  let bee_A : Bee := ⟨2, 3, 1, 0, 0⟩
  let bee_B : Bee := ⟨0, 0, 0, 4, 1⟩
  let bee_C : Bee := ⟨1, 2, 2, 3, 1⟩
  let bees : List Bee := [bee_A, bee_B, bee_C]

  total_flowers = roses + sunflowers + tulips + daisies + orchids ∧
  (bees.map (λ b => b.roses_per_hour + b.sunflowers_per_hour + b.tulips_per_hour + 
                    b.daisies_per_hour + b.orchids_per_hour)).sum * hours = 60 ∧
  ∀ ft : FlowerType, 
    (bees.map (λ b => match ft with
      | FlowerType.Rose => b.roses_per_hour
      | FlowerType.Sunflower => b.sunflowers_per_hour
      | FlowerType.Tulip => b.tulips_per_hour
      | FlowerType.Daisy => b.daisies_per_hour
      | FlowerType.Orchid => b.orchids_per_hour
    )).sum * hours ≤ match ft with
      | FlowerType.Rose => roses
      | FlowerType.Sunflower => sunflowers
      | FlowerType.Tulip => tulips
      | FlowerType.Daisy => daisies
      | FlowerType.Orchid => orchids

theorem flower_pollination : flower_problem := by sorry

end NUMINAMATH_CALUDE_flower_pollination_l3540_354007


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3540_354021

-- Define the equation of the parabolas
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 3) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 4)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  parabola_equation vertex1.1 vertex1.2 ∧
  parabola_equation vertex2.1 vertex2.2 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_vertices_l3540_354021


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3540_354058

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3540_354058


namespace NUMINAMATH_CALUDE_floor_properties_l3540_354030

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_properties (x y : ℝ) :
  (x - 1 < floor x) ∧
  (floor x - floor y - 1 < x - y) ∧
  (x - y < floor x - floor y + 1) ∧
  (x^2 + 1/3 > floor x) :=
by sorry

end NUMINAMATH_CALUDE_floor_properties_l3540_354030


namespace NUMINAMATH_CALUDE_magnitude_sum_of_vectors_l3540_354076

/-- Given two plane vectors a and b, prove that |a + b| = √5 under specific conditions -/
theorem magnitude_sum_of_vectors (a b : ℝ × ℝ) : 
  a = (1, 1) → 
  ‖b‖ = 1 → 
  Real.cos (Real.pi / 4) * ‖a‖ * ‖b‖ = a.fst * b.fst + a.snd * b.snd →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_sum_of_vectors_l3540_354076


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3540_354069

/-- Given a rectangle ABCD and a square EFGH, if 30% of the rectangle's area
    overlaps with the square, and 25% of the square's area overlaps with the rectangle,
    then the ratio of the rectangle's length to its width is 7.5. -/
theorem rectangle_square_overlap_ratio :
  ∀ (l w s : ℝ),
    l > 0 → w > 0 → s > 0 →
    (0.3 * l * w = 0.25 * s^2) →
    (l / w = 7.5) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3540_354069


namespace NUMINAMATH_CALUDE_honda_cars_in_chennai_l3540_354022

def total_cars : ℕ := 900
def red_honda_percentage : ℚ := 90 / 100
def total_red_percentage : ℚ := 60 / 100
def red_non_honda_percentage : ℚ := 225 / 1000

theorem honda_cars_in_chennai :
  ∃ (h : ℕ), h = 500 ∧
  (h : ℚ) * red_honda_percentage + (total_cars - h : ℚ) * red_non_honda_percentage = (total_cars : ℚ) * total_red_percentage :=
by sorry

end NUMINAMATH_CALUDE_honda_cars_in_chennai_l3540_354022


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3540_354015

theorem polynomial_factorization (a b : ℤ) :
  (∀ x : ℝ, 24 * x^2 - 158 * x - 147 = (12 * x + a) * (2 * x + b)) →
  a + 2 * b = -35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3540_354015


namespace NUMINAMATH_CALUDE_cat_ratio_l3540_354084

theorem cat_ratio (jacob_cats : ℕ) (melanie_cats : ℕ) :
  jacob_cats = 90 →
  melanie_cats = 60 →
  ∃ (annie_cats : ℕ),
    annie_cats = jacob_cats / 3 ∧
    melanie_cats = annie_cats ∧
    melanie_cats / annie_cats = 2 :=
by sorry

end NUMINAMATH_CALUDE_cat_ratio_l3540_354084


namespace NUMINAMATH_CALUDE_instantaneous_acceleration_at_3s_l3540_354037

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Define the acceleration function as the derivative of velocity
def acceleration (t : ℝ) : ℝ := 12 * t

-- Theorem statement
theorem instantaneous_acceleration_at_3s :
  acceleration 3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_instantaneous_acceleration_at_3s_l3540_354037


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l3540_354051

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles * (red_marbles - 1) * blue_marbles * (blue_marbles - 1)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem two_red_two_blue_probability :
  probability_two_red_two_blue = 1232 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l3540_354051


namespace NUMINAMATH_CALUDE_population_difference_after_increase_l3540_354004

/-- Represents the population of birds in a wildlife reserve -/
structure BirdPopulation where
  eagles : ℕ
  falcons : ℕ
  hawks : ℕ
  owls : ℕ

/-- Calculates the difference between the most and least populous bird types -/
def populationDifference (pop : BirdPopulation) : ℕ :=
  max pop.eagles (max pop.falcons (max pop.hawks pop.owls)) -
  min pop.eagles (min pop.falcons (min pop.hawks pop.owls))

/-- Calculates the new population after increasing the least populous by 10% -/
def increaseLeastPopulous (pop : BirdPopulation) : BirdPopulation :=
  let minPop := min pop.eagles (min pop.falcons (min pop.hawks pop.owls))
  let increase := minPop * 10 / 100
  { eagles := if pop.eagles = minPop then pop.eagles + increase else pop.eagles,
    falcons := if pop.falcons = minPop then pop.falcons + increase else pop.falcons,
    hawks := if pop.hawks = minPop then pop.hawks + increase else pop.hawks,
    owls := if pop.owls = minPop then pop.owls + increase else pop.owls }

theorem population_difference_after_increase (initialPop : BirdPopulation) :
  initialPop.eagles = 150 →
  initialPop.falcons = 200 →
  initialPop.hawks = 320 →
  initialPop.owls = 270 →
  populationDifference (increaseLeastPopulous initialPop) = 155 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_after_increase_l3540_354004


namespace NUMINAMATH_CALUDE_sqrt_2023_divided_by_sum_of_digits_l3540_354057

theorem sqrt_2023_divided_by_sum_of_digits : Real.sqrt (2023 / (2 + 0 + 2 + 3)) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_divided_by_sum_of_digits_l3540_354057


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l3540_354098

theorem jose_bottle_caps (initial : Real) (given_away : Real) (remaining : Real) : 
  initial = 7.0 → given_away = 2.0 → remaining = initial - given_away → remaining = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l3540_354098


namespace NUMINAMATH_CALUDE_gcd_problem_l3540_354085

theorem gcd_problem :
  ∃! n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd n 15 = 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3540_354085


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_twelve_l3540_354023

theorem twenty_percent_greater_than_twelve (x : ℝ) : 
  x = 12 * (1 + 0.2) → x = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_twelve_l3540_354023


namespace NUMINAMATH_CALUDE_rope_for_first_post_l3540_354094

theorem rope_for_first_post (second_post third_post fourth_post total : ℕ) 
  (h1 : second_post = 20)
  (h2 : third_post = 14)
  (h3 : fourth_post = 12)
  (h4 : total = 70)
  (h5 : ∃ first_post : ℕ, first_post + second_post + third_post + fourth_post = total) :
  ∃ first_post : ℕ, first_post = 24 ∧ first_post + second_post + third_post + fourth_post = total :=
by
  sorry

end NUMINAMATH_CALUDE_rope_for_first_post_l3540_354094


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3540_354074

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₃a₅ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  a 4 = 4 →                                        -- given condition
  a 3 * a 5 = 16 :=                                -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3540_354074


namespace NUMINAMATH_CALUDE_restaurant_expenditure_l3540_354028

theorem restaurant_expenditure (total_people : Nat) (standard_spenders : Nat) (standard_amount : ℝ) (total_spent : ℝ) :
  total_people = 8 →
  standard_spenders = 7 →
  standard_amount = 10 →
  total_spent = 88 →
  (total_spent - (standard_spenders * standard_amount)) - (total_spent / total_people) = 7 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_expenditure_l3540_354028


namespace NUMINAMATH_CALUDE_ivar_water_planning_l3540_354065

def water_planning (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (total_water : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption := total_horses * (drinking_water + bathing_water)
  total_water / daily_consumption

theorem ivar_water_planning :
  water_planning 3 5 5 2 1568 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ivar_water_planning_l3540_354065


namespace NUMINAMATH_CALUDE_number_of_pupils_in_class_l3540_354041

/-- 
Given a class where:
1. A pupil's marks were wrongly entered as 67 instead of 45.
2. The wrong entry caused the average marks for the class to increase by half a mark.

Prove that the number of pupils in the class is 44.
-/
theorem number_of_pupils_in_class : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_in_class_l3540_354041


namespace NUMINAMATH_CALUDE_probability_three_heads_twelve_coins_l3540_354006

theorem probability_three_heads_twelve_coins : 
  (Nat.choose 12 3 : ℚ) / (2^12 : ℚ) = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_twelve_coins_l3540_354006


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3540_354088

theorem fahrenheit_to_celsius (F C : ℚ) : F = (9 / 5) * C + 32 → F = 10 → C = -110 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3540_354088


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l3540_354014

/-- The height of the pole in meters -/
def pole_height : ℝ := 10

/-- The time taken to reach the top of the pole in minutes -/
def total_time : ℕ := 17

/-- The distance the monkey slips in alternate minutes -/
def slip_distance : ℝ := 1

/-- The distance the monkey ascends in the first minute -/
def ascend_distance : ℝ := 1.8

/-- The number of complete ascend-slip cycles -/
def num_cycles : ℕ := (total_time - 1) / 2

theorem monkey_climb_theorem :
  ascend_distance + num_cycles * (ascend_distance - slip_distance) + ascend_distance = pole_height :=
sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l3540_354014


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3540_354050

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x + 2 * Real.sqrt y)^5 = a₀*x^5 + a₁*x^4*(Real.sqrt y) + a₂*x^3*y + a₃*x^2*y*(Real.sqrt y) + a₄*x*y^2 + a₅*y^(5/2)) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3540_354050


namespace NUMINAMATH_CALUDE_age_fraction_proof_l3540_354087

theorem age_fraction_proof (age : ℕ) (h : age = 64) :
  (8 * (age + 8) - 8 * (age - 8)) / age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_fraction_proof_l3540_354087


namespace NUMINAMATH_CALUDE_zeros_of_continuous_function_l3540_354016

theorem zeros_of_continuous_function (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cont : Continuous f) 
  (h_order : a < b ∧ b < c) 
  (h_sign1 : f a * f b < 0) 
  (h_sign2 : f b * f c < 0) : 
  ∃ (n : ℕ), n ≥ 2 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_continuous_function_l3540_354016


namespace NUMINAMATH_CALUDE_f_domain_l3540_354066

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

theorem f_domain : Set.Icc (-1 : ℝ) 1 = {x : ℝ | ∃ y, f x = y} :=
sorry

end NUMINAMATH_CALUDE_f_domain_l3540_354066


namespace NUMINAMATH_CALUDE_percentage_problem_l3540_354039

theorem percentage_problem (N P : ℝ) : 
  N = 50 → 
  N = (P / 100) * N + 40 → 
  P = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3540_354039


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l3540_354053

/-- Represents the face of a coin -/
inductive CoinFace
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle := List CoinFace

/-- Checks if two adjacent coins have the same face -/
def hasAdjacentSameFace (circle : CoinCircle) : Prop :=
  ∃ i, (circle.get? i = circle.get? ((i + 1) % circle.length))

/-- Theorem: Any arrangement of 11 coins in a circle always has at least one pair of adjacent coins with the same face -/
theorem adjacent_same_face_exists (circle : CoinCircle) (h : circle.length = 11) :
  hasAdjacentSameFace circle :=
sorry

end NUMINAMATH_CALUDE_adjacent_same_face_exists_l3540_354053


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3540_354040

theorem sum_of_fractions : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3540_354040


namespace NUMINAMATH_CALUDE_apple_juice_percentage_is_40_percent_l3540_354068

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  apples : Nat
  appleJuice : Nat
  bananas : Nat
  bananaJuice : Nat

/-- Calculates the percentage of apple juice in a blend -/
def appleJuicePercentage (yield : JuiceYield) : Rat :=
  let appleJuicePerFruit := yield.appleJuice / yield.apples
  let bananaJuicePerFruit := yield.bananaJuice / yield.bananas
  let totalJuice := appleJuicePerFruit + bananaJuicePerFruit
  appleJuicePerFruit / totalJuice

/-- Theorem: The percentage of apple juice in the blend is 40% -/
theorem apple_juice_percentage_is_40_percent (yield : JuiceYield) 
    (h1 : yield.apples = 5)
    (h2 : yield.appleJuice = 10)
    (h3 : yield.bananas = 4)
    (h4 : yield.bananaJuice = 12) : 
  appleJuicePercentage yield = 2/5 := by
  sorry

#eval (2 : Rat) / 5

end NUMINAMATH_CALUDE_apple_juice_percentage_is_40_percent_l3540_354068


namespace NUMINAMATH_CALUDE_min_value_of_a_l3540_354090

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := 
  fun x => if x > 0 then Real.exp x + a else -(Real.exp (-x) + a)

-- State the theorem
theorem min_value_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) →  -- f is odd
  (∀ x y : ℝ, x < y → f a x < f a y) →  -- f is strictly increasing (monotonic)
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x < f b y) → 
            b ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3540_354090


namespace NUMINAMATH_CALUDE_trick_sum_prediction_l3540_354018

theorem trick_sum_prediction (a b : ℕ) (ha : 10000 ≤ a ∧ a < 100000) : 
  a + b + (99999 - b) = 100000 + a - 1 := by
  sorry

end NUMINAMATH_CALUDE_trick_sum_prediction_l3540_354018


namespace NUMINAMATH_CALUDE_expression_factorization_l3540_354002

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 45 * x^2 - 10) - (-5 * x^3 + 15 * x^2 - 5) = 5 * (5 * x^3 + 6 * x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3540_354002


namespace NUMINAMATH_CALUDE_linear_function_properties_l3540_354005

def f (x : ℝ) : ℝ := -2 * x + 3

theorem linear_function_properties :
  (f 1 = 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f⁻¹ 0 ≠ 0) ∧
  (∀ (x1 x2 : ℝ), x1 < x2 → f x1 > f x2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3540_354005


namespace NUMINAMATH_CALUDE_two_colonies_limit_days_l3540_354093

/-- Represents the number of days it takes for a single bacteria colony to reach the habitat limit -/
def single_colony_limit_days : ℕ := 20

/-- Represents the growth rate of the bacteria colony (doubling every day) -/
def growth_rate : ℚ := 2

/-- Represents the fixed habitat limit -/
def habitat_limit : ℚ := growth_rate ^ single_colony_limit_days

/-- Theorem stating that two colonies reach the habitat limit in the same number of days as one colony -/
theorem two_colonies_limit_days (initial_colonies : ℕ) (h : initial_colonies = 2) :
  (initial_colonies * growth_rate ^ single_colony_limit_days = habitat_limit) :=
sorry

end NUMINAMATH_CALUDE_two_colonies_limit_days_l3540_354093


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l3540_354081

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem monotonicity_and_extrema :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1)) → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = 2) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = -18) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l3540_354081


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3540_354035

theorem quadratic_inequality_empty_solution : 
  {x : ℝ | -x^2 + 2*x - 3 > 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3540_354035


namespace NUMINAMATH_CALUDE_odometer_reading_before_trip_l3540_354049

theorem odometer_reading_before_trip 
  (odometer_at_lunch : ℝ) 
  (miles_traveled : ℝ) 
  (h1 : odometer_at_lunch = 372.0)
  (h2 : miles_traveled = 159.7) :
  odometer_at_lunch - miles_traveled = 212.3 := by
sorry

end NUMINAMATH_CALUDE_odometer_reading_before_trip_l3540_354049


namespace NUMINAMATH_CALUDE_trapezoid_area_l3540_354080

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (BC : ℝ)
  (AD : ℝ)
  (AB : ℝ)
  (CD : ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the given trapezoid is 59 -/
theorem trapezoid_area :
  let t : Trapezoid := { BC := 9.5, AD := 20, AB := 5, CD := 8.5 }
  area t = 59 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3540_354080


namespace NUMINAMATH_CALUDE_expression_values_l3540_354082

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d > 0) :
  let expr := a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| + d / |d|
  expr = 5 ∨ expr = 1 ∨ expr = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3540_354082


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3540_354063

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let x₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  let x₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  x₁ + x₂ = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3540_354063


namespace NUMINAMATH_CALUDE_remainder_481207_div_8_l3540_354062

theorem remainder_481207_div_8 :
  ∃ q : ℕ, 481207 = 8 * q + 7 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_481207_div_8_l3540_354062


namespace NUMINAMATH_CALUDE_no_lions_present_l3540_354077

theorem no_lions_present (total : ℕ) (tigers monkeys : ℕ) : 
  tigers = 7 * (total - tigers) →
  monkeys = (total - monkeys) / 7 →
  tigers + monkeys = total →
  ∀ other : ℕ, other ≤ total - (tigers + monkeys) → other = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_lions_present_l3540_354077


namespace NUMINAMATH_CALUDE_correct_calculation_l3540_354042

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3540_354042


namespace NUMINAMATH_CALUDE_divisibility_property_l3540_354036

theorem divisibility_property (a b c d : ℤ) 
  (h : (a - c) ∣ (a * b + c * d)) : 
  (a - c) ∣ (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3540_354036


namespace NUMINAMATH_CALUDE_problem_statement_l3540_354013

theorem problem_statement (a b c : ℝ) 
  (h1 : -10 ≤ a) (h2 : a < 0) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3540_354013


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l3540_354092

open Real

theorem tangent_equation_solution (x : ℝ) :
  (5.44 * tan (5 * x) - 2 * tan (3 * x) = tan (3 * x)^2 * tan (5 * x)) →
  (cos (3 * x) ≠ 0) →
  (cos (5 * x) ≠ 0) →
  ∃ k : ℤ, x = π * k := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l3540_354092


namespace NUMINAMATH_CALUDE_square_ge_of_ge_pos_l3540_354027

theorem square_ge_of_ge_pos {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_ge_of_ge_pos_l3540_354027


namespace NUMINAMATH_CALUDE_interpretation_correct_l3540_354091

-- Define propositions
variable (p : Prop)  -- Student A's math score is not less than 100 points
variable (q : Prop)  -- Student B's math score is less than 100 points

-- Define the interpretation of p∨(¬q)
def interpretation : Prop := p ∨ (¬q)

-- Theorem statement
theorem interpretation_correct : 
  interpretation p q ↔ (p ∨ ¬q) :=
sorry

end NUMINAMATH_CALUDE_interpretation_correct_l3540_354091


namespace NUMINAMATH_CALUDE_find_divisor_l3540_354009

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 919 →
  quotient = 17 →
  remainder = 11 →
  dividend = divisor * quotient + remainder →
  divisor = 53 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3540_354009


namespace NUMINAMATH_CALUDE_unique_positive_n_l3540_354026

/-- A quadratic equation has exactly one real root if and only if its discriminant is zero -/
axiom discriminant_zero_iff_one_root {a b c : ℝ} (ha : a ≠ 0) :
  b^2 - 4*a*c = 0 ↔ ∃! x, a*x^2 + b*x + c = 0

/-- The quadratic equation y^2 + 6ny + 9n has exactly one real root -/
def has_one_root (n : ℝ) : Prop :=
  ∃! y, y^2 + 6*n*y + 9*n = 0

theorem unique_positive_n :
  ∃! n : ℝ, n > 0 ∧ has_one_root n :=
sorry

end NUMINAMATH_CALUDE_unique_positive_n_l3540_354026


namespace NUMINAMATH_CALUDE_james_monthly_income_l3540_354001

/-- Represents a subscription tier with its subscriber count and price --/
structure SubscriptionTier where
  subscribers : ℕ
  price : ℚ

/-- Calculates the total monthly income for James from Twitch subscriptions --/
def calculate_monthly_income (tier1 tier2 tier3 : SubscriptionTier) : ℚ :=
  tier1.subscribers * tier1.price +
  tier2.subscribers * tier2.price +
  tier3.subscribers * tier3.price

/-- Theorem stating that James' monthly income from Twitch subscriptions is $2522.50 --/
theorem james_monthly_income :
  let tier1 := SubscriptionTier.mk (120 + 10) (499 / 100)
  let tier2 := SubscriptionTier.mk (50 + 25) (999 / 100)
  let tier3 := SubscriptionTier.mk (30 + 15) (2499 / 100)
  calculate_monthly_income tier1 tier2 tier3 = 252250 / 100 := by
  sorry


end NUMINAMATH_CALUDE_james_monthly_income_l3540_354001


namespace NUMINAMATH_CALUDE_class_test_probabilities_l3540_354064

theorem class_test_probabilities (P_A P_B P_neither : ℝ)
  (h_A : P_A = 0.8)
  (h_B : P_B = 0.55)
  (h_neither : P_neither = 0.55) :
  P_A + P_B - (1 - P_neither) = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_class_test_probabilities_l3540_354064


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3540_354071

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_equals_interval : S ∩ T = Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3540_354071


namespace NUMINAMATH_CALUDE_min_xy_value_l3540_354034

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3*x*y - x - y - 1 = 0) :
  ∀ z, z = x*y → z ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3540_354034


namespace NUMINAMATH_CALUDE_evaluate_expression_l3540_354070

theorem evaluate_expression : (3^2)^3 + 2*(3^2 - 2^3) = 731 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3540_354070


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_47_l3540_354017

def is_multiple (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem least_positive_integer_multiple_47 :
  ∃! x : ℕ+, (x : ℤ) = 5 ∧ 
  (∀ y : ℕ+, y < x → ¬ is_multiple ((2 * y : ℤ)^2 + 2 * 37 * (2 * y) + 37^2) 47) ∧
  is_multiple ((2 * x : ℤ)^2 + 2 * 37 * (2 * x) + 37^2) 47 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_47_l3540_354017


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3540_354099

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3540_354099


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3540_354095

theorem quadratic_root_theorem (a b c d : ℝ) (h : a ≠ 0) :
  (a * (b - c - d) * 1^2 + b * (c - a + d) * 1 + c * (a - b - d) = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ 
    a * (b - c - d) * x^2 + b * (c - a + d) * x + c * (a - b - d) = 0 ∧
    x = c * (a - b - d) / (a * (b - c - d)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3540_354095


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3540_354054

-- Define the set of positive integers
def PositiveInt := {n : ℤ | n > 0}

-- Define the functional equation
def SatisfiesEquation (f : ℚ → ℤ) : Prop :=
  ∀ (x : ℚ) (a : ℤ) (b : PositiveInt), f ((f x + a) / b) = f ((x + a) / b)

-- Define the possible solution functions
def ConstantFunction (C : ℤ) : ℚ → ℤ := λ _ => C
def FloorFunction : ℚ → ℤ := λ x => ⌊x⌋
def CeilingFunction : ℚ → ℤ := λ x => ⌈x⌉

-- State the theorem
theorem functional_equation_solutions (f : ℚ → ℤ) (h : SatisfiesEquation f) :
  (∃ C : ℤ, f = ConstantFunction C) ∨ f = FloorFunction ∨ f = CeilingFunction :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3540_354054


namespace NUMINAMATH_CALUDE_max_value_of_ab_l3540_354055

theorem max_value_of_ab (a b : ℝ) : 
  (Real.sqrt 3 = Real.sqrt (3^a * 3^b)) → (∀ x y : ℝ, (Real.sqrt 3 = Real.sqrt (3^x * 3^y)) → a * b ≥ x * y) → 
  a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l3540_354055


namespace NUMINAMATH_CALUDE_school_election_votes_l3540_354011

theorem school_election_votes (bob_votes : ℕ) (total_votes : ℕ) 
  (h1 : bob_votes = 48)
  (h2 : bob_votes = (2 : ℕ) * total_votes / (5 : ℕ)) :
  total_votes = 120 := by
  sorry

end NUMINAMATH_CALUDE_school_election_votes_l3540_354011


namespace NUMINAMATH_CALUDE_greatest_integer_pi_minus_five_l3540_354000

theorem greatest_integer_pi_minus_five :
  ⌊Real.pi - 5⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_pi_minus_five_l3540_354000


namespace NUMINAMATH_CALUDE_systematic_sampling_correct_l3540_354067

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  start : ℕ

/-- Generates the sequence of selected student numbers -/
def generate_sequence (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (fun i => s.start + i * (s.total_students / s.sample_size))

/-- Theorem: The systematic sampling of 6 students from 60 results in the correct sequence -/
theorem systematic_sampling_correct : 
  let s : SystematicSampling := ⟨60, 6, 6⟩
  generate_sequence s = [6, 16, 26, 36, 46, 56] := by
  sorry

#eval generate_sequence ⟨60, 6, 6⟩

end NUMINAMATH_CALUDE_systematic_sampling_correct_l3540_354067


namespace NUMINAMATH_CALUDE_right_triangle_properties_l3540_354043

/-- A right triangle with sides 9 cm and 12 cm -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area : ℝ
  side1_eq : side1 = 9
  side2_eq : side2 = 12
  pythagorean : side1^2 + side2^2 = hypotenuse^2
  area_formula : area = (1/2) * side1 * side2

/-- The hypotenuse of the right triangle is 15 cm and its area is 54 cm² -/
theorem right_triangle_properties (t : RightTriangle) : t.hypotenuse = 15 ∧ t.area = 54 := by
  sorry

#check right_triangle_properties

end NUMINAMATH_CALUDE_right_triangle_properties_l3540_354043


namespace NUMINAMATH_CALUDE_quiz_scores_l3540_354047

theorem quiz_scores (nicole kim cherry : ℕ) 
  (h1 : nicole = kim - 3)
  (h2 : kim = cherry + 8)
  (h3 : nicole = 22) : 
  cherry = 17 := by sorry

end NUMINAMATH_CALUDE_quiz_scores_l3540_354047


namespace NUMINAMATH_CALUDE_expected_value_of_win_l3540_354020

def fair_8_sided_die := Finset.range 8

def win_amount (n : ℕ) : ℝ := 8 - n

theorem expected_value_of_win :
  Finset.sum fair_8_sided_die (λ n => (1 : ℝ) / 8 * win_amount n) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_win_l3540_354020


namespace NUMINAMATH_CALUDE_square_diff_eq_three_implies_product_eq_nine_l3540_354048

theorem square_diff_eq_three_implies_product_eq_nine (x y : ℝ) :
  x^2 - y^2 = 3 → (x + y)^2 * (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_eq_three_implies_product_eq_nine_l3540_354048


namespace NUMINAMATH_CALUDE_probability_theorem_l3540_354079

/-- A rectangle with dimensions 3 × 2 units -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- 10 points evenly spaced along the perimeter of the rectangle -/
def num_points : ℕ := 10

/-- The probability of selecting two points one unit apart -/
def probability_one_unit_apart (rect : Rectangle) : ℚ :=
  2 / 9

/-- Theorem stating the probability of selecting two points one unit apart -/
theorem probability_theorem (rect : Rectangle) 
  (h1 : rect.length = 3) 
  (h2 : rect.width = 2) : 
  probability_one_unit_apart rect = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3540_354079


namespace NUMINAMATH_CALUDE_investment_amount_correct_l3540_354010

/-- Calculates the investment amount in T-shirt printing equipment -/
def calculate_investment (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) : ℚ :=
  selling_price * break_even_point - cost_per_shirt * break_even_point

/-- Proves that the investment amount is correct -/
theorem investment_amount_correct (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) :
  calculate_investment cost_per_shirt selling_price break_even_point = 1411 :=
by
  have h1 : cost_per_shirt = 3 := by sorry
  have h2 : selling_price = 20 := by sorry
  have h3 : break_even_point = 83 := by sorry
  sorry

#eval calculate_investment 3 20 83

end NUMINAMATH_CALUDE_investment_amount_correct_l3540_354010
