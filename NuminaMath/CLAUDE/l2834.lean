import Mathlib

namespace NUMINAMATH_CALUDE_kindergarten_tissues_l2834_283470

/-- The number of tissues brought by kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l2834_283470


namespace NUMINAMATH_CALUDE_panthers_second_half_score_l2834_283468

theorem panthers_second_half_score 
  (total_first_half : ℕ)
  (cougars_lead_first_half : ℕ)
  (total_game : ℕ)
  (cougars_lead_total : ℕ)
  (h1 : total_first_half = 38)
  (h2 : cougars_lead_first_half = 16)
  (h3 : total_game = 58)
  (h4 : cougars_lead_total = 22) :
  ∃ (cougars_first cougars_second panthers_first panthers_second : ℕ),
    cougars_first + panthers_first = total_first_half ∧
    cougars_first = panthers_first + cougars_lead_first_half ∧
    cougars_first + cougars_second + panthers_first + panthers_second = total_game ∧
    (cougars_first + cougars_second) - (panthers_first + panthers_second) = cougars_lead_total ∧
    panthers_second = 7 :=
by sorry

end NUMINAMATH_CALUDE_panthers_second_half_score_l2834_283468


namespace NUMINAMATH_CALUDE_total_cost_is_30_l2834_283452

def silverware_cost : ℝ := 20
def dinner_plates_cost_ratio : ℝ := 0.5

def total_cost : ℝ :=
  silverware_cost + (silverware_cost * dinner_plates_cost_ratio)

theorem total_cost_is_30 :
  total_cost = 30 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_30_l2834_283452


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2834_283469

def A (θ : Real) : Set Real := {1, Real.sin θ}
def B : Set Real := {1/2, 2}

theorem sufficient_but_not_necessary :
  (∀ θ : Real, θ = 5 * Real.pi / 6 → A θ ∩ B = {1/2}) ∧
  (∃ θ : Real, θ ≠ 5 * Real.pi / 6 ∧ A θ ∩ B = {1/2}) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2834_283469


namespace NUMINAMATH_CALUDE_max_crate_weight_l2834_283495

/-- Proves that the maximum weight each crate can hold is 20 kg given the problem conditions --/
theorem max_crate_weight (num_crates : ℕ) (nail_bags : ℕ) (hammer_bags : ℕ) (plank_bags : ℕ)
  (nail_weight : ℝ) (hammer_weight : ℝ) (plank_weight : ℝ) (left_out_weight : ℝ) :
  num_crates = 15 →
  nail_bags = 4 →
  hammer_bags = 12 →
  plank_bags = 10 →
  nail_weight = 5 →
  hammer_weight = 5 →
  plank_weight = 30 →
  left_out_weight = 80 →
  (nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight - left_out_weight) / num_crates = 20 := by
  sorry

#check max_crate_weight

end NUMINAMATH_CALUDE_max_crate_weight_l2834_283495


namespace NUMINAMATH_CALUDE_exists_x_where_exp_leq_x_plus_one_l2834_283457

theorem exists_x_where_exp_leq_x_plus_one : ∃ x : ℝ, Real.exp x ≤ x + 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_where_exp_leq_x_plus_one_l2834_283457


namespace NUMINAMATH_CALUDE_marty_stripes_l2834_283442

/-- The number of narrow black stripes on Marty the zebra -/
def narrow_black_stripes : ℕ := 8

/-- The number of wide black stripes on Marty the zebra -/
def wide_black_stripes : ℕ := sorry

/-- The number of white stripes on Marty the zebra -/
def white_stripes : ℕ := wide_black_stripes + 7

/-- The total number of black stripes on Marty the zebra -/
def total_black_stripes : ℕ := wide_black_stripes + narrow_black_stripes

theorem marty_stripes : 
  total_black_stripes = white_stripes + 1 → 
  narrow_black_stripes = 8 := by
  sorry

end NUMINAMATH_CALUDE_marty_stripes_l2834_283442


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2834_283486

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4)) ∧ 
  x = 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2834_283486


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l2834_283425

theorem ceiling_sqrt_count : 
  (Finset.range 226 \ Finset.range 197).card = 29 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l2834_283425


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2834_283446

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence with first term -6 and common difference 5 -/
theorem twentieth_term_of_sequence :
  arithmeticSequence (-6) 5 20 = 89 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2834_283446


namespace NUMINAMATH_CALUDE_triangle_ABC_point_C_l2834_283476

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem triangle_ABC_point_C :
  ∀ (C : ℝ × ℝ),
  (C.2 = line_C C.1) →  -- C lies on the line y = 3x + 3
  (abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2 = triangle_area) →  -- Area of triangle ABC is 10
  (C = (-1, 0) ∨ C = (5/3, 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_point_C_l2834_283476


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2834_283463

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 40) :
  (perimeter / 4) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2834_283463


namespace NUMINAMATH_CALUDE_five_congruent_subtriangles_impossible_l2834_283402

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a subdivision of a triangle into five smaller triangles
structure SubdividedTriangle where
  main : Triangle
  sub1 : Triangle
  sub2 : Triangle
  sub3 : Triangle
  sub4 : Triangle
  sub5 : Triangle

-- Theorem statement
theorem five_congruent_subtriangles_impossible (t : SubdividedTriangle) :
  ¬(t.sub1 = t.sub2 ∧ t.sub2 = t.sub3 ∧ t.sub3 = t.sub4 ∧ t.sub4 = t.sub5) :=
by sorry

end NUMINAMATH_CALUDE_five_congruent_subtriangles_impossible_l2834_283402


namespace NUMINAMATH_CALUDE_infinitely_many_n_exist_l2834_283445

-- Define the s operation on sets of integers
def s (F : Set ℤ) : Set ℤ :=
  {a : ℤ | (a ∈ F ∧ a - 1 ∉ F) ∨ (a ∉ F ∧ a - 1 ∈ F)}

-- Define the n-fold application of s
def s_power (F : Set ℤ) : ℕ → Set ℤ
  | 0 => F
  | n + 1 => s (s_power F n)

theorem infinitely_many_n_exist (F : Set ℤ) (h_finite : Set.Finite F) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, s_power F n = F ∪ {a + n | a ∈ F} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_exist_l2834_283445


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2834_283412

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2834_283412


namespace NUMINAMATH_CALUDE_num_technicians_correct_l2834_283459

/-- The number of technicians in a workshop with given conditions. -/
def num_technicians : ℕ :=
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  7

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem num_technicians_correct :
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  let num_technicians := num_technicians
  let num_rest := total_workers - num_technicians
  (num_technicians * avg_salary_technicians + num_rest * avg_salary_rest) / total_workers = avg_salary_all :=
by
  sorry

#eval num_technicians

end NUMINAMATH_CALUDE_num_technicians_correct_l2834_283459


namespace NUMINAMATH_CALUDE_convergence_of_iterative_process_l2834_283492

theorem convergence_of_iterative_process (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, 2^(-k : ℤ) * (a - b) < (1 : ℝ) / 2002 := by
  sorry

end NUMINAMATH_CALUDE_convergence_of_iterative_process_l2834_283492


namespace NUMINAMATH_CALUDE_swimming_passings_l2834_283415

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  duration : ℝ

/-- Calculates the number of times swimmers pass each other -/
def calculatePassings (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of passings in the given scenario -/
theorem swimming_passings :
  let scenario : SwimmingScenario := {
    poolLength := 100,
    swimmerASpeed := 4,
    swimmerBSpeed := 5,
    duration := 30 * 60  -- 30 minutes in seconds
  }
  calculatePassings scenario = 54 := by sorry

end NUMINAMATH_CALUDE_swimming_passings_l2834_283415


namespace NUMINAMATH_CALUDE_A_star_B_equality_l2834_283456

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x ≥ 1}

def star_operation (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

theorem A_star_B_equality : 
  star_operation A B = {x : ℝ | (0 ≤ x ∧ x < 1) ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_A_star_B_equality_l2834_283456


namespace NUMINAMATH_CALUDE_total_students_count_l2834_283467

/-- Represents the number of students who scored 60 marks -/
def x : ℕ := sorry

/-- The total number of students in the class -/
def total_students : ℕ := 10 + 15 + x

/-- The average marks for the whole class -/
def class_average : ℕ := 72

/-- The theorem stating the total number of students in the class -/
theorem total_students_count : total_students = 50 := by
  have h1 : (10 * 90 + 15 * 80 + x * 60) / total_students = class_average := by sorry
  sorry

end NUMINAMATH_CALUDE_total_students_count_l2834_283467


namespace NUMINAMATH_CALUDE_calculate_expression_l2834_283422

theorem calculate_expression : (1/2)⁻¹ + |Real.sqrt 3 - 2| + Real.sqrt 12 = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2834_283422


namespace NUMINAMATH_CALUDE_winning_percentage_l2834_283418

/-- Given an election with 6000 total votes and a winning margin of 1200 votes,
    prove that the winning candidate received 60% of the votes. -/
theorem winning_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) :
  total_votes = 6000 →
  winning_margin = 1200 →
  winning_percentage = 60 / 100 →
  winning_percentage * total_votes = (total_votes + winning_margin) / 2 :=
by sorry

end NUMINAMATH_CALUDE_winning_percentage_l2834_283418


namespace NUMINAMATH_CALUDE_no_solution_equation_l2834_283432

theorem no_solution_equation :
  ¬ ∃ x : ℝ, (x + 2) / (x - 2) - x / (x + 2) = 16 / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2834_283432


namespace NUMINAMATH_CALUDE_some_number_value_l2834_283420

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (x / 3) = 61 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2834_283420


namespace NUMINAMATH_CALUDE_camping_matches_l2834_283443

def matches_left (initial : ℕ) (dropped : ℕ) : ℕ :=
  initial - dropped - 2 * dropped

theorem camping_matches (initial : ℕ) (dropped : ℕ) 
  (h1 : initial ≥ dropped) 
  (h2 : initial ≥ dropped + 2 * dropped) :
  matches_left initial dropped = initial - dropped - 2 * dropped :=
by sorry

end NUMINAMATH_CALUDE_camping_matches_l2834_283443


namespace NUMINAMATH_CALUDE_polynomial_composition_l2834_283437

/-- Given a function f and a polynomial g, proves that g satisfies the given condition -/
theorem polynomial_composition (f g : ℝ → ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, f (g x) = 4*x^2 + 4*x + 1) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_l2834_283437


namespace NUMINAMATH_CALUDE_train_departure_time_l2834_283451

/-- Proves that the first train left Mumbai 2 hours before the meeting point -/
theorem train_departure_time 
  (first_train_speed : ℝ) 
  (second_train_speed : ℝ) 
  (time_difference : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : first_train_speed = 45)
  (h2 : second_train_speed = 90)
  (h3 : time_difference = 1)
  (h4 : meeting_distance = 90) :
  ∃ (departure_time : ℝ), 
    departure_time = 2 ∧ 
    first_train_speed * (departure_time + time_difference) = 
    second_train_speed * time_difference ∧
    first_train_speed * departure_time = meeting_distance :=
by sorry


end NUMINAMATH_CALUDE_train_departure_time_l2834_283451


namespace NUMINAMATH_CALUDE_definite_integral_abs_x_squared_minus_two_l2834_283461

theorem definite_integral_abs_x_squared_minus_two :
  ∫ x in (-2)..1, |x^2 - 2| = 1/3 + 8*Real.sqrt 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_abs_x_squared_minus_two_l2834_283461


namespace NUMINAMATH_CALUDE_john_needs_more_money_l2834_283427

theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) (h1 : total_needed = 2.50) (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_more_money_l2834_283427


namespace NUMINAMATH_CALUDE_sector_angle_measure_l2834_283493

/-- Given a circular sector with radius 10 and area 50π/3, 
    prove that its central angle measures π/3 radians. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (α : ℝ) 
  (h_radius : r = 10)
  (h_area : S = 50 * Real.pi / 3)
  (h_sector_area : S = 1/2 * r^2 * α) :
  α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l2834_283493


namespace NUMINAMATH_CALUDE_product_correction_l2834_283453

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem product_correction (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 → -- a is a three-digit number
  (reverseDigits a) * b = 468 → -- incorrect calculation
  a * b = 1116 := by sorry

end NUMINAMATH_CALUDE_product_correction_l2834_283453


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2834_283423

theorem arithmetic_calculation : 8 - 7 + 6 * 5 + 4 - 3 * 2 + 1 - 0 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2834_283423


namespace NUMINAMATH_CALUDE_expression_simplification_l2834_283434

theorem expression_simplification :
  let x := (1 : ℝ) / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2)))
  let y := (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / ((9 * Real.sqrt 5 + 4 * Real.sqrt 7)^2 - 100)
  x = y := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2834_283434


namespace NUMINAMATH_CALUDE_parabola_vertex_on_negative_x_axis_l2834_283401

/-- Given a parabola y = x^2 - bx + 8, if its vertex lies on the negative half-axis of the x-axis, then b = -4√2 -/
theorem parabola_vertex_on_negative_x_axis (b : ℝ) :
  (∃ x, x < 0 ∧ x^2 - b*x + 8 = 0 ∧ ∀ y, y ≠ x → (y^2 - b*y + 8 > 0)) →
  b = -4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_negative_x_axis_l2834_283401


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2834_283466

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 : ℝ)^2 + 5*(2 : ℝ) + m = 0 → m = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2834_283466


namespace NUMINAMATH_CALUDE_rational_reachability_l2834_283465

-- Define the operations
def f (x : ℚ) : ℚ := (1 + x) / x
def g (x : ℚ) : ℚ := (1 - x) / x

-- Define a type for sequences of operations
inductive Op
| F : Op
| G : Op

def apply_op (op : Op) (x : ℚ) : ℚ :=
  match op with
  | Op.F => f x
  | Op.G => g x

def apply_ops (ops : List Op) (x : ℚ) : ℚ :=
  ops.foldl (λ acc op => apply_op op acc) x

theorem rational_reachability (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (ops : List Op), apply_ops ops a = b :=
sorry

end NUMINAMATH_CALUDE_rational_reachability_l2834_283465


namespace NUMINAMATH_CALUDE_no_common_solution_exists_l2834_283448

/-- A_{n}^k denotes the number of k-permutations of n elements -/
def A (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- C_{n}^k denotes the number of k-combinations of n elements -/
def C (n k : ℕ) : ℕ := Nat.choose n k

theorem no_common_solution_exists : ¬ ∃ (n : ℕ), n ≥ 3 ∧ 
  A (2*n) 3 = 2 * A (n+1) 4 ∧ 
  C (n+2) (n-2) + C (n+2) (n-3) = (A (n+3) 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_exists_l2834_283448


namespace NUMINAMATH_CALUDE_card_collection_difference_l2834_283473

/-- Represents the number of cards each person has -/
structure CardCollection where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ
  carla : ℕ
  desmond : ℕ

/-- The conditions of the card collection problem -/
def card_collection_conditions (c : CardCollection) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.carla = 4 * c.heike ∧
  c.desmond = 8 * c.heike ∧
  c.ann = 60

/-- The theorem stating the difference between the highest and lowest number of cards -/
theorem card_collection_difference (c : CardCollection) 
  (h : card_collection_conditions c) : 
  max c.anton (max c.ann (max c.bertrand (max c.carla c.desmond))) - 
  min c.heike (min c.anton (min c.ann (min c.bertrand (min c.carla c.desmond)))) = 70 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_difference_l2834_283473


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2834_283414

theorem quadratic_equation_solution (a b : ℕ+) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + 12*x = 73 ∧ x = Real.sqrt a - b) → a + b = 115 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2834_283414


namespace NUMINAMATH_CALUDE_diamond_five_three_l2834_283438

-- Define the operation ⋄
def diamond (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem statement
theorem diamond_five_three : diamond 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_diamond_five_three_l2834_283438


namespace NUMINAMATH_CALUDE_water_volume_is_fifty_l2834_283405

/-- A cubical tank partially filled with water -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_fraction : ℝ

/-- The volume of water in the tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.capacity_fraction * tank.side_length^3

theorem water_volume_is_fifty (tank : CubicalTank) 
  (h1 : tank.water_level = 2)
  (h2 : tank.capacity_fraction = 0.4)
  (h3 : tank.water_level = tank.capacity_fraction * tank.side_length) :
  water_volume tank = 50 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_is_fifty_l2834_283405


namespace NUMINAMATH_CALUDE_farm_field_area_l2834_283411

/-- The area of a farm field given specific ploughing conditions --/
theorem farm_field_area (planned_rate : ℝ) (actual_rate : ℝ) (extra_days : ℕ) (area_left : ℝ) : 
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  ∃ (planned_days : ℝ), 
    planned_rate * planned_days = actual_rate * (planned_days + extra_days) + area_left ∧
    planned_rate * planned_days = 720 := by
  sorry

end NUMINAMATH_CALUDE_farm_field_area_l2834_283411


namespace NUMINAMATH_CALUDE_cube_surface_area_l2834_283482

/-- The surface area of a cube with side length 8 cm is 384 cm². -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length^2
  surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2834_283482


namespace NUMINAMATH_CALUDE_age_difference_is_54_l2834_283404

/-- Represents a person's age with tens and units digits -/
structure Age where
  tens : Nat
  units : Nat
  tens_nonzero : tens ≠ 0

/-- The problem statement -/
theorem age_difference_is_54 
  (jack : Age) 
  (bill : Age) 
  (h1 : jack.tens * 10 + jack.units + 10 = 3 * (bill.tens * 10 + bill.units + 10))
  (h2 : jack.tens = bill.units ∧ jack.units = bill.tens) :
  (jack.tens * 10 + jack.units) - (bill.tens * 10 + bill.units) = 54 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_54_l2834_283404


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_bound_l2834_283428

/-- A triangle in a 2D plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle --/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half the area of the triangle --/
theorem inscribed_rectangle_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_bound_l2834_283428


namespace NUMINAMATH_CALUDE_square_root_possible_value_l2834_283424

theorem square_root_possible_value (a : ℝ) : 
  (a = -1 ∨ a = -6 ∨ a = 3 ∨ a = -7) → 
  (∃ x : ℝ, x^2 = a) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_root_possible_value_l2834_283424


namespace NUMINAMATH_CALUDE_two_digit_factorizations_of_2079_l2834_283481

/-- A factorization of a number into two factors -/
structure Factorization :=
  (factor1 : ℕ)
  (factor2 : ℕ)

/-- Check if a number is two-digit (between 10 and 99, inclusive) -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Check if a factorization is valid for 2079 with two-digit factors -/
def isValidFactorization (f : Factorization) : Prop :=
  f.factor1 * f.factor2 = 2079 ∧ isTwoDigit f.factor1 ∧ isTwoDigit f.factor2

/-- Two factorizations are considered equal if they have the same factors (in any order) -/
def factorizationEqual (f1 f2 : Factorization) : Prop :=
  (f1.factor1 = f2.factor1 ∧ f1.factor2 = f2.factor2) ∨
  (f1.factor1 = f2.factor2 ∧ f1.factor2 = f2.factor1)

/-- The main theorem: there are exactly 2 unique factorizations of 2079 into two-digit numbers -/
theorem two_digit_factorizations_of_2079 :
  ∃ (f1 f2 : Factorization),
    isValidFactorization f1 ∧
    isValidFactorization f2 ∧
    ¬factorizationEqual f1 f2 ∧
    ∀ (f : Factorization), isValidFactorization f → (factorizationEqual f f1 ∨ factorizationEqual f f2) :=
  sorry

end NUMINAMATH_CALUDE_two_digit_factorizations_of_2079_l2834_283481


namespace NUMINAMATH_CALUDE_pen_purchase_shortfall_l2834_283413

/-- The amount of money needed to purchase a pen given the cost, initial amount, and borrowed amount -/
theorem pen_purchase_shortfall (pen_cost : ℕ) (initial_amount : ℕ) (borrowed_amount : ℕ) :
  pen_cost = 600 →
  initial_amount = 500 →
  borrowed_amount = 68 →
  pen_cost - (initial_amount + borrowed_amount) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pen_purchase_shortfall_l2834_283413


namespace NUMINAMATH_CALUDE_complex_point_in_second_quadrant_l2834_283458

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem complex_point_in_second_quadrant (a : ℝ) (z : ℂ) 
  (h1 : z = a + Complex.I) 
  (h2 : Complex.abs z < Real.sqrt 2) : 
  is_in_second_quadrant (a - 1) 1 := by
  sorry

#check complex_point_in_second_quadrant

end NUMINAMATH_CALUDE_complex_point_in_second_quadrant_l2834_283458


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_11_16_l2834_283496

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℝ
  edge_positive : edge_length > 0

/-- Represents the larger cube constructed from smaller cubes -/
structure LargeCube where
  cube : Cube
  small_cubes : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  black_corners : ℕ
  black_face_centers : ℕ

/-- Calculates the fraction of white surface area for the large cube -/
def white_surface_fraction (lc : LargeCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem white_surface_fraction_is_11_16 :
  let lc : LargeCube := {
    cube := { edge_length := 4, edge_positive := by norm_num },
    small_cubes := 64,
    black_cubes := 24,
    white_cubes := 40,
    black_corners := 8,
    black_face_centers := 6
  }
  white_surface_fraction lc = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_11_16_l2834_283496


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2834_283485

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x + 2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  (4 : ℝ) * x - y - 1 = 0 ↔ y - y₀ = m * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2834_283485


namespace NUMINAMATH_CALUDE_four_pockets_sixteen_coins_l2834_283491

/-- The total number of coins in multiple pockets -/
def total_coins (num_pockets : ℕ) (coins_per_pocket : ℕ) : ℕ :=
  num_pockets * coins_per_pocket

/-- Theorem: Given 4 pockets with 16 coins each, the total number of coins is 64 -/
theorem four_pockets_sixteen_coins : total_coins 4 16 = 64 := by
  sorry

end NUMINAMATH_CALUDE_four_pockets_sixteen_coins_l2834_283491


namespace NUMINAMATH_CALUDE_triangle_angle_side_difference_l2834_283455

theorem triangle_angle_side_difference (y : ℝ) : 
  (y + 6 > 0) →  -- AB > 0
  (y + 3 > 0) →  -- AC > 0
  (2*y > 0) →    -- BC > 0
  (y + 6 + y + 3 > 2*y) →  -- AB + AC > BC
  (y + 6 + 2*y > y + 3) →  -- AB + BC > AC
  (y + 3 + 2*y > y + 6) →  -- AC + BC > AB
  (2*y > y + 6) →          -- BC > AB (for ∠A to be largest)
  (2*y > y + 3) →          -- BC > AC (for ∠A to be largest)
  (max (y + 6) (y + 3) - min (y + 6) (y + 3) ≥ 3) ∧ 
  (∃ (y : ℝ), max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_side_difference_l2834_283455


namespace NUMINAMATH_CALUDE_percent_problem_l2834_283499

theorem percent_problem (x : ℝ) (h : 4 = 0.08 * x) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2834_283499


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2834_283426

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2834_283426


namespace NUMINAMATH_CALUDE_expected_heads_theorem_l2834_283478

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The number of maximum flips -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads on a single flip -/
def prob_heads : ℚ := 1/2

/-- The probability of a coin showing heads after all flips -/
def prob_heads_after_flips : ℚ := 15/16

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_flips

theorem expected_heads_theorem :
  ⌊expected_heads⌋ = 94 :=
sorry

end NUMINAMATH_CALUDE_expected_heads_theorem_l2834_283478


namespace NUMINAMATH_CALUDE_eight_person_arrangement_l2834_283409

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangements (n : ℕ) (a b c : ℕ) : ℕ :=
  factorial n - (factorial (n-1) * 2) - (factorial (n-2) * 6 - factorial (n-1) * 2)

theorem eight_person_arrangement : arrangements 8 1 1 1 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_eight_person_arrangement_l2834_283409


namespace NUMINAMATH_CALUDE_swimming_club_van_capacity_l2834_283439

/-- Calculates the maximum capacity of each van given the conditions of the swimming club problem --/
theorem swimming_club_van_capacity 
  (num_cars : ℕ) 
  (num_vans : ℕ) 
  (people_per_car : ℕ) 
  (people_per_van : ℕ) 
  (max_car_capacity : ℕ) 
  (additional_capacity : ℕ) 
  (h1 : num_cars = 2)
  (h2 : num_vans = 3)
  (h3 : people_per_car = 5)
  (h4 : people_per_van = 3)
  (h5 : max_car_capacity = 6)
  (h6 : additional_capacity = 17) :
  (num_cars * max_car_capacity + num_vans * 
    ((num_cars * people_per_car + num_vans * people_per_van + additional_capacity) / num_vans - 
     num_cars * max_car_capacity / num_vans)) / num_vans = 8 := by
  sorry

#check swimming_club_van_capacity

end NUMINAMATH_CALUDE_swimming_club_van_capacity_l2834_283439


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2009_l2834_283480

/-- The units digit of 3^n for any natural number n -/
def unitsDigitOf3Pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

theorem units_digit_of_3_pow_2009 :
  unitsDigitOf3Pow 2009 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2009_l2834_283480


namespace NUMINAMATH_CALUDE_stating_distribution_schemes_eq_60_l2834_283449

/-- Represents the number of female students -/
def num_female : ℕ := 5

/-- Represents the number of male students -/
def num_male : ℕ := 2

/-- Represents the number of groups -/
def num_groups : ℕ := 2

/-- 
Calculates the number of ways to distribute students into groups
such that each group has at least one female and one male student
-/
def distribution_schemes (f : ℕ) (m : ℕ) (g : ℕ) : ℕ :=
  2 * (2^f - 2)

/-- 
Theorem stating that the number of distribution schemes
for the given problem is 60
-/
theorem distribution_schemes_eq_60 :
  distribution_schemes num_female num_male num_groups = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_distribution_schemes_eq_60_l2834_283449


namespace NUMINAMATH_CALUDE_smallest_n_value_l2834_283460

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def quadruplet_count : ℕ := 84000

/-- The given GCD value -/
def gcd_value : ℕ := 84

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    satisfying gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n that satisfies the conditions -/
theorem smallest_n_value : 
  (∀ m < 1555848, count_quadruplets m ≠ quadruplet_count) ∧ 
  count_quadruplets 1555848 = quadruplet_count := by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2834_283460


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2834_283498

/-- An isosceles triangle with two sides measuring 5 and 12 has a perimeter of 29. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 5 ∧ c = 12 ∨ a = 12 ∧ c = 5)) →
  a + b + c = 29 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2834_283498


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2834_283433

/-- A regular polygon with interior angles of 160 degrees and side length of 4 units has 18 sides and a perimeter of 72 units. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (side_length : ℝ),
    n > 2 →
    side_length = 4 →
    (180 * (n - 2) : ℝ) / n = 160 →
    n = 18 ∧ n * side_length = 72 := by
  sorry

#check regular_polygon_properties

end NUMINAMATH_CALUDE_regular_polygon_properties_l2834_283433


namespace NUMINAMATH_CALUDE_chinese_number_puzzle_l2834_283410

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem chinese_number_puzzle :
  ∀ (x y z : ℕ),
    x < 100 →
    y < 10000 →
    z < 10000 →
    100 * x + y + z = 2015 →
    z + sum_n 10 = y →
    x ≠ (y / 1000) →
    x ≠ (y / 100 % 10) →
    x ≠ (y / 10 % 10) →
    x ≠ (y % 10) →
    x ≠ (z / 1000) →
    x ≠ (z / 100 % 10) →
    x ≠ (z / 10 % 10) →
    x ≠ (z % 10) →
    (y / 1000) ≠ (y / 100 % 10) →
    (y / 1000) ≠ (y / 10 % 10) →
    (y / 1000) ≠ (y % 10) →
    (y / 100 % 10) ≠ (y / 10 % 10) →
    (y / 100 % 10) ≠ (y % 10) →
    (y / 10 % 10) ≠ (y % 10) →
    (z / 1000) ≠ (z / 100 % 10) →
    (z / 1000) ≠ (z / 10 % 10) →
    (z / 1000) ≠ (z % 10) →
    (z / 100 % 10) ≠ (z / 10 % 10) →
    (z / 100 % 10) ≠ (z % 10) →
    (z / 10 % 10) ≠ (z % 10) →
    100 * x + y = 1985 :=
by
  sorry

#eval sum_n 10  -- This should evaluate to 55

end NUMINAMATH_CALUDE_chinese_number_puzzle_l2834_283410


namespace NUMINAMATH_CALUDE_wire_cut_square_circle_ratio_l2834_283479

theorem wire_cut_square_circle_ratio (x y : ℝ) (h : x > 0) (k : y > 0) : 
  (x^2 / 16 = y^2 / (4 * Real.pi)) → x / y = 2 / Real.sqrt Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_circle_ratio_l2834_283479


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l2834_283484

theorem tailor_cut_difference (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → pants_cut = 0.5 → skirt_cut - pants_cut = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l2834_283484


namespace NUMINAMATH_CALUDE_prop_p_iff_prop_q_l2834_283483

theorem prop_p_iff_prop_q (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ m) ↔
  (∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_prop_p_iff_prop_q_l2834_283483


namespace NUMINAMATH_CALUDE_jordan_running_time_l2834_283494

/-- Given that Jordan ran 4 miles in one-third the time it took Steve to run 6 miles,
    and Steve took 36 minutes to run 6 miles, prove that Jordan would take 21 minutes
    to run 7 miles. -/
theorem jordan_running_time
  (steve_time : ℝ)
  (steve_distance : ℝ)
  (jordan_distance : ℝ)
  (jordan_time_fraction : ℝ)
  (jordan_new_distance : ℝ)
  (h1 : steve_time = 36)
  (h2 : steve_distance = 6)
  (h3 : jordan_distance = 4)
  (h4 : jordan_time_fraction = 1 / 3)
  (h5 : jordan_new_distance = 7)
  : (jordan_new_distance * jordan_time_fraction * steve_time) / jordan_distance = 21 := by
  sorry

#check jordan_running_time

end NUMINAMATH_CALUDE_jordan_running_time_l2834_283494


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_two_l2834_283487

def point_on_number_line (x : ℝ) := True

theorem point_three_units_from_negative_two (A : ℝ) :
  point_on_number_line A →
  |A - (-2)| = 3 →
  A = -5 ∨ A = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_two_l2834_283487


namespace NUMINAMATH_CALUDE_stratified_sampling_car_inspection_l2834_283475

theorem stratified_sampling_car_inspection
  (total_sample : ℕ)
  (type_a_production type_b_production type_c_production : ℕ)
  (h_total_sample : total_sample = 47)
  (h_type_a : type_a_production = 1400)
  (h_type_b : type_b_production = 6000)
  (h_type_c : type_c_production = 2000) :
  ∃ (sample_a sample_b sample_c : ℕ),
    sample_a + sample_b + sample_c = total_sample ∧
    sample_a = 7 ∧
    sample_b = 30 ∧
    sample_c = 10 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_car_inspection_l2834_283475


namespace NUMINAMATH_CALUDE_baf_compound_composition_l2834_283440

/-- Represents the molecular structure of a compound containing Barium and Fluorine --/
structure BaFCompound where
  ba_count : ℕ
  f_count : ℕ
  molecular_weight : ℝ

/-- Atomic weights of elements --/
def atomic_weight : String → ℝ
  | "Ba" => 137.33
  | "F" => 18.998
  | _ => 0

/-- Calculates the molecular weight of a BaFCompound --/
def calculate_weight (c : BaFCompound) : ℝ :=
  c.ba_count * atomic_weight "Ba" + c.f_count * atomic_weight "F"

/-- Theorem stating that a compound with 2 Fluorine atoms and molecular weight 175 contains 1 Barium atom --/
theorem baf_compound_composition :
  ∃ (c : BaFCompound), c.f_count = 2 ∧ c.molecular_weight = 175 ∧ c.ba_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_baf_compound_composition_l2834_283440


namespace NUMINAMATH_CALUDE_january_oil_bill_l2834_283454

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 30) / january_bill = 5 / 3) →
  january_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l2834_283454


namespace NUMINAMATH_CALUDE_five_numbers_satisfy_conditions_l2834_283472

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  (n.val / 10) + (n.val % 10)

/-- Performs the operation described in the problem -/
def operation (n : TwoDigitNumber) : ℕ :=
  n.val - sumOfDigits n

/-- Checks if the units digit of a number is 4 -/
def hasUnitsDigit4 (n : ℕ) : Prop :=
  n % 10 = 4

/-- The main theorem stating that exactly 5 two-digit numbers satisfy the conditions -/
theorem five_numbers_satisfy_conditions :
  ∃! (s : Finset TwoDigitNumber),
    (∀ n ∈ s, isEven n.val ∧ hasUnitsDigit4 (operation n)) ∧
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_five_numbers_satisfy_conditions_l2834_283472


namespace NUMINAMATH_CALUDE_building_occupancy_l2834_283471

/-- Given a building with a certain number of stories, apartments per floor, and people per apartment,
    calculate the total number of people housed in the building. -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people per apartment
    houses 200 people in total. -/
theorem building_occupancy :
  total_people 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_occupancy_l2834_283471


namespace NUMINAMATH_CALUDE_graveling_cost_l2834_283403

/-- The cost of graveling two intersecting roads on a rectangular lawn -/
theorem graveling_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 4 →
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sqm = 5200 := by
  sorry

end NUMINAMATH_CALUDE_graveling_cost_l2834_283403


namespace NUMINAMATH_CALUDE_reflect_P_across_y_axis_l2834_283436

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(1, -2) across the y-axis results in (-1, -2) -/
theorem reflect_P_across_y_axis :
  let P : Point := { x := 1, y := -2 }
  reflectAcrossYAxis P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_y_axis_l2834_283436


namespace NUMINAMATH_CALUDE_lcm_54_75_l2834_283488

theorem lcm_54_75 : Nat.lcm 54 75 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_54_75_l2834_283488


namespace NUMINAMATH_CALUDE_additional_miles_for_average_speed_l2834_283435

theorem additional_miles_for_average_speed 
  (initial_distance : ℝ) 
  (initial_speed : ℝ) 
  (desired_average_speed : ℝ) 
  (additional_speed : ℝ) : 
  initial_distance = 20 ∧ 
  initial_speed = 40 ∧ 
  desired_average_speed = 55 ∧ 
  additional_speed = 60 → 
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / 
    (initial_distance / initial_speed + additional_distance / additional_speed) = 
    desired_average_speed ∧ 
    additional_distance = 90 := by
sorry

end NUMINAMATH_CALUDE_additional_miles_for_average_speed_l2834_283435


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l2834_283417

/-- Given a quadratic equation y = ax^2 + bx + c, 
    this theorem proves that 2a - b + c = 2 
    when a = 2, b = 3, and c = 1 -/
theorem quadratic_equation_value (a b c : ℝ) : 
  a = 2 ∧ b = 3 ∧ c = 1 → 2*a - b + c = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l2834_283417


namespace NUMINAMATH_CALUDE_fibonacci_fifth_is_s_plus_one_l2834_283430

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def s : ℕ := 4

theorem fibonacci_fifth_is_s_plus_one :
  fibonacci 5 = s + 1 ∧ ∀ k < 5, fibonacci k ≠ s + 1 := by sorry

end NUMINAMATH_CALUDE_fibonacci_fifth_is_s_plus_one_l2834_283430


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2834_283497

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2 * x - y) + (3 * x - 2 * y) = 2 * x - y := by sorry

-- Problem 2
theorem evaluate_expression : 
  -(1^4) + |3 - 5| - 8 + (-2) * (1/2) = -8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2834_283497


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2834_283489

def vec_a : ℝ × ℝ := (2, 3)
def vec_b : ℝ × ℝ := (-1, 5)

theorem vector_addition_scalar_multiplication :
  vec_a + 3 • vec_b = (-1, 18) := by sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2834_283489


namespace NUMINAMATH_CALUDE_secret_spread_reaches_target_l2834_283447

/-- Represents the number of students who know the secret on a given day -/
def secret_spread : ℕ → ℕ
| 0 => 4  -- Monday (day 0): Jessica + 3 friends
| 1 => 10 -- Tuesday (day 1): Previous + 2 * 3 new
| 2 => 22 -- Wednesday (day 2): Previous + 3 * 3 + 3 new
| n + 3 => secret_spread (n + 2) + 3 * (secret_spread (n + 2) - secret_spread (n + 1))

/-- The day when the secret reaches at least 7280 students -/
def target_day : ℕ := 9

theorem secret_spread_reaches_target :
  secret_spread target_day ≥ 7280 := by
  sorry


end NUMINAMATH_CALUDE_secret_spread_reaches_target_l2834_283447


namespace NUMINAMATH_CALUDE_hot_dog_stand_sales_l2834_283444

/-- A hot dog stand problem -/
theorem hot_dog_stand_sales 
  (price : ℝ) 
  (hours : ℝ) 
  (total_sales : ℝ) 
  (h1 : price = 2)
  (h2 : hours = 10)
  (h3 : total_sales = 200) :
  total_sales / (hours * price) = 10 :=
sorry

end NUMINAMATH_CALUDE_hot_dog_stand_sales_l2834_283444


namespace NUMINAMATH_CALUDE_monotone_sine_range_l2834_283431

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/4, 2π/3] if and only if ω is in (0, 3/4] -/
theorem monotone_sine_range (ω : ℝ) (h : ω > 0) :
  StrictMonoOn (fun x => 2 * Real.sin (ω * x)) (Set.Icc (-π/4) (2*π/3)) ↔ ω ∈ Set.Ioo 0 (3/4) ∪ {3/4} := by
  sorry

end NUMINAMATH_CALUDE_monotone_sine_range_l2834_283431


namespace NUMINAMATH_CALUDE_root_in_interval_l2834_283474

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem root_in_interval :
  ∃! r : ℝ, r ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2834_283474


namespace NUMINAMATH_CALUDE_min_value_theorem_l2834_283429

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  (3 / x) + (1 / (y - 3)) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ + 3 * x₀ = 3 ∧ 0 < x₀ ∧ x₀ < 1/2 ∧ (3 / x₀) + (1 / (y₀ - 3)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2834_283429


namespace NUMINAMATH_CALUDE_problem_solvers_equal_girls_l2834_283464

/-- Given a class of students, prove that the number of students who solved a problem
    is equal to the total number of girls, given that the number of boys who solved
    the problem is equal to the number of girls who did not solve it. -/
theorem problem_solvers_equal_girls (total : ℕ) (boys girls : ℕ) 
    (boys_solved girls_solved : ℕ) : 
    boys + girls = total →
    boys_solved = girls - girls_solved →
    boys_solved + girls_solved = girls := by
  sorry

end NUMINAMATH_CALUDE_problem_solvers_equal_girls_l2834_283464


namespace NUMINAMATH_CALUDE_x_polynomial_equality_l2834_283450

theorem x_polynomial_equality (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + x^4 = 35292*x - 13652 := by
  sorry

end NUMINAMATH_CALUDE_x_polynomial_equality_l2834_283450


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2834_283477

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 479160 → n + (n + 1) + (n + 2) = 234 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2834_283477


namespace NUMINAMATH_CALUDE_letter_150_is_Z_l2834_283441

/-- Represents the letters in the repeating pattern -/
inductive Letter
| X
| Y
| Z

/-- The length of the repeating pattern -/
def pattern_length : Nat := 3

/-- Function to determine the nth letter in the repeating pattern -/
def nth_letter (n : Nat) : Letter :=
  match n % pattern_length with
  | 0 => Letter.Z
  | 1 => Letter.X
  | _ => Letter.Y

/-- Theorem stating that the 150th letter in the pattern is Z -/
theorem letter_150_is_Z : nth_letter 150 = Letter.Z := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_Z_l2834_283441


namespace NUMINAMATH_CALUDE_min_sum_squares_l2834_283421

theorem min_sum_squares (a b c : ℕ+) (h : a.val^2 + b.val^2 - c.val = 2022) :
  (∀ a' b' c' : ℕ+, a'.val^2 + b'.val^2 - c'.val = 2022 →
    a.val^2 + b.val^2 + c.val^2 ≤ a'.val^2 + b'.val^2 + c'.val^2) ∧
  a.val^2 + b.val^2 + c.val^2 = 2034 ∧
  a.val = 27 ∧ b.val = 36 ∧ c.val = 3 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2834_283421


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2834_283406

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence a with a₁ = 1 and a₅ = 16, prove that a₃ = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = 1) 
    (h_a5 : a 5 = 16) : 
  a 3 = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l2834_283406


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2834_283416

/-- The y-intercept of the line 5x - 3y = 15 is (0, -5) -/
theorem y_intercept_of_line (x y : ℝ) :
  5 * x - 3 * y = 15 → y = -5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2834_283416


namespace NUMINAMATH_CALUDE_equation_solution_unique_solution_l2834_283400

theorem equation_solution : ∃ (x : ℝ), x = 2 ∧ -2 * x + 4 = 0 := by
  sorry

-- Definitions of the given equations
def eq1 (x : ℝ) : Prop := 3 * x + 6 = 0
def eq2 (x : ℝ) : Prop := -2 * x + 4 = 0
def eq3 (x : ℝ) : Prop := (1 / 2) * x = 2
def eq4 (x : ℝ) : Prop := 2 * x + 4 = 0

-- Theorem stating that eq2 is the only equation satisfied by x = 2
theorem unique_solution :
  ∃! (i : Fin 4), (match i with
    | 0 => eq1
    | 1 => eq2
    | 2 => eq3
    | 3 => eq4) 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_solution_l2834_283400


namespace NUMINAMATH_CALUDE_complex_power_patterns_l2834_283408

theorem complex_power_patterns (i : ℂ) (h : i^2 = -1) :
  ∀ n : ℕ,
    i^(4*n + 1) = i ∧
    i^(4*n + 2) = -1 ∧
    i^(4*n + 3) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_power_patterns_l2834_283408


namespace NUMINAMATH_CALUDE_complex_trig_simplification_l2834_283490

open Complex

theorem complex_trig_simplification (θ : ℝ) :
  let z₁ := (cos θ - I * sin θ) ^ 8
  let z₂ := (1 + I * tan θ) ^ 5
  let z₃ := (cos θ + I * sin θ) ^ 2
  let z₄ := tan θ + I
  (z₁ * z₂) / (z₃ * z₄) = -((sin (4 * θ) + I * cos (4 * θ)) / (cos θ) ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_trig_simplification_l2834_283490


namespace NUMINAMATH_CALUDE_house_construction_bricks_house_construction_bricks_specific_l2834_283419

/-- Calculates the number of bricks needed for house construction given specific costs and requirements. -/
theorem house_construction_bricks (land_cost_per_sqm : ℕ) (brick_cost_per_thousand : ℕ) 
  (roof_tile_cost : ℕ) (land_area : ℕ) (roof_tiles : ℕ) (total_cost : ℕ) : ℕ :=
  let land_cost := land_cost_per_sqm * land_area
  let roof_cost := roof_tile_cost * roof_tiles
  let brick_budget := total_cost - land_cost - roof_cost
  let bricks_thousands := brick_budget / brick_cost_per_thousand
  bricks_thousands * 1000

/-- Proves that given the specific conditions, the number of bricks needed is 10,000. -/
theorem house_construction_bricks_specific : 
  house_construction_bricks 50 100 10 2000 500 106000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_house_construction_bricks_house_construction_bricks_specific_l2834_283419


namespace NUMINAMATH_CALUDE_trig_identity_for_point_l2834_283462

/-- Given a point P on the terminal side of angle α with coordinates (4a, -3a) where a < 0,
    prove that 2sin(α) + cos(α) = 2/5 -/
theorem trig_identity_for_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_trig_identity_for_point_l2834_283462


namespace NUMINAMATH_CALUDE_victors_work_hours_l2834_283407

theorem victors_work_hours (hourly_rate : ℝ) (total_earnings : ℝ) (h : ℝ) : 
  hourly_rate = 6 → 
  total_earnings = 60 → 
  2 * (hourly_rate * h) = total_earnings → 
  h = 5 := by
sorry

end NUMINAMATH_CALUDE_victors_work_hours_l2834_283407
