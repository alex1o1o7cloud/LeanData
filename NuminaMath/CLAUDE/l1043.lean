import Mathlib

namespace NUMINAMATH_CALUDE_max_correct_answers_l1043_104385

/-- Represents an exam score. -/
structure ExamScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  totalQuestions : ℕ
  score : ℤ

/-- Checks if the exam score is valid according to the rules. -/
def ExamScore.isValid (e : ExamScore) : Prop :=
  e.correct + e.incorrect + e.unanswered = e.totalQuestions ∧
  6 * e.correct - 3 * e.incorrect = e.score

/-- Theorem: The maximum number of correct answers for the given exam conditions is 14. -/
theorem max_correct_answers :
  ∀ e : ExamScore,
    e.totalQuestions = 25 →
    e.score = 57 →
    e.isValid →
    e.correct ≤ 14 ∧
    ∃ e' : ExamScore, e'.totalQuestions = 25 ∧ e'.score = 57 ∧ e'.isValid ∧ e'.correct = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1043_104385


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1043_104358

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, 13 = y^2 + 1/y^2 ∧ y + 1/y = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1043_104358


namespace NUMINAMATH_CALUDE_min_rb_selling_price_theorem_l1043_104313

/-- Represents the fruit sales problem -/
structure FruitSales where
  total_weight : ℝ
  total_cost : ℝ
  rb_purchase_price : ℝ
  rb_selling_price_last_week : ℝ
  xg_purchase_price : ℝ
  xg_selling_price : ℝ
  rb_damage_rate : ℝ

/-- Calculates the profit from last week's sales -/
def last_week_profit (fs : FruitSales) : ℝ := sorry

/-- Calculates the minimum selling price for Red Beauty this week -/
def min_rb_selling_price_this_week (fs : FruitSales) : ℝ := sorry

/-- Theorem stating the minimum selling price of Red Beauty this week -/
theorem min_rb_selling_price_theorem (fs : FruitSales) 
  (h1 : fs.total_weight = 300)
  (h2 : fs.total_cost = 3000)
  (h3 : fs.rb_purchase_price = 20)
  (h4 : fs.rb_selling_price_last_week = 35)
  (h5 : fs.xg_purchase_price = 5)
  (h6 : fs.xg_selling_price = 10)
  (h7 : fs.rb_damage_rate = 0.1)
  : min_rb_selling_price_this_week fs ≥ 36.7 ∧ 
    last_week_profit fs = 2500 := by sorry

end NUMINAMATH_CALUDE_min_rb_selling_price_theorem_l1043_104313


namespace NUMINAMATH_CALUDE_corporation_employees_l1043_104396

theorem corporation_employees (part_time : ℕ) (full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_corporation_employees_l1043_104396


namespace NUMINAMATH_CALUDE_dog_arrangement_theorem_l1043_104319

theorem dog_arrangement_theorem (n : ℕ) (h : n = 5) :
  (n! / 2) = 60 :=
sorry

end NUMINAMATH_CALUDE_dog_arrangement_theorem_l1043_104319


namespace NUMINAMATH_CALUDE_composite_sequence_l1043_104386

theorem composite_sequence (a n : ℕ) (ha : a ≥ 2) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (a^k + i).Prime = false :=
sorry

end NUMINAMATH_CALUDE_composite_sequence_l1043_104386


namespace NUMINAMATH_CALUDE_inequality_proof_l1043_104391

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + 3*c) / (3*a + 3*b + 2*c) + 
  (a + 3*b + c) / (3*a + 2*b + 3*c) + 
  (3*a + b + c) / (2*a + 3*b + 3*c) ≥ 15/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1043_104391


namespace NUMINAMATH_CALUDE_wednesday_water_intake_total_water_intake_correct_l1043_104323

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Water intake for a given day -/
def water_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 9
  | Day.Tuesday => 8
  | Day.Wednesday => 9  -- This is what we want to prove
  | Day.Thursday => 9
  | Day.Friday => 8
  | Day.Saturday => 9
  | Day.Sunday => 8

/-- Total water intake for the week -/
def total_water_intake : ℕ := 60

/-- Theorem: The water intake on Wednesday is 9 liters -/
theorem wednesday_water_intake :
  water_intake Day.Wednesday = 9 :=
by
  sorry

/-- Theorem: The total water intake for the week is correct -/
theorem total_water_intake_correct :
  (water_intake Day.Monday) +
  (water_intake Day.Tuesday) +
  (water_intake Day.Wednesday) +
  (water_intake Day.Thursday) +
  (water_intake Day.Friday) +
  (water_intake Day.Saturday) +
  (water_intake Day.Sunday) = total_water_intake :=
by
  sorry

end NUMINAMATH_CALUDE_wednesday_water_intake_total_water_intake_correct_l1043_104323


namespace NUMINAMATH_CALUDE_parabola_equation_l1043_104393

/-- A parabola with directrix x = 1/2 has the standard equation y^2 = -2x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = 1/2) →  -- directrix is x = 1/2
  (∃! f : ℝ × ℝ → Prop, ∀ x y, f (x, y) ↔ y^2 = -2*x) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1043_104393


namespace NUMINAMATH_CALUDE_henrys_age_l1043_104334

theorem henrys_age (h s : ℕ) : 
  h + 8 = 3 * (s - 1) →
  (h - 25) + (s - 25) = 83 →
  h = 97 :=
by sorry

end NUMINAMATH_CALUDE_henrys_age_l1043_104334


namespace NUMINAMATH_CALUDE_worker_b_days_l1043_104370

/-- The number of days it takes for worker b to complete a job alone,
    given that worker a is twice as efficient as worker b and
    together they complete the job in 6 days. -/
theorem worker_b_days (efficiency_a : ℝ) (efficiency_b : ℝ) (days_together : ℝ) : 
  efficiency_a = 2 * efficiency_b →
  days_together = 6 →
  efficiency_a + efficiency_b = 1 / days_together →
  1 / efficiency_b = 18 := by
sorry

end NUMINAMATH_CALUDE_worker_b_days_l1043_104370


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l1043_104357

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane if they intersect at right angles -/
def line_perp_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel if they do not intersect -/
def planes_parallel (α β : Plane) : Prop := sorry

theorem line_perp_parallel_planes (α β : Plane) (m : Line) :
  different_planes α β →
  line_perp_plane m β →
  planes_parallel α β →
  line_perp_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l1043_104357


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1043_104352

/-- The volume of ice cream in a right circular cone topped with a hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = 48 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l1043_104352


namespace NUMINAMATH_CALUDE_weight_loss_days_l1043_104394

/-- Calculates the number of days required to lose a target weight given daily calorie intake, burn rate, and calories per pound of weight loss. -/
def daysToLoseWeight (caloriesEaten : ℕ) (caloriesBurned : ℕ) (caloriesPerPound : ℕ) (targetPounds : ℕ) : ℕ :=
  (caloriesPerPound * targetPounds) / (caloriesBurned - caloriesEaten)

theorem weight_loss_days :
  daysToLoseWeight 1800 2300 4000 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_days_l1043_104394


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1043_104345

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.3)
  let total_decrease_percent := 0.37
  let second_year_decrease_percent := (initial_value * total_decrease_percent - (initial_value - first_year_value)) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1043_104345


namespace NUMINAMATH_CALUDE_village_population_l1043_104362

theorem village_population (population_percentage : Real) (partial_population : Nat) :
  population_percentage = 80 →
  partial_population = 23040 →
  (partial_population : Real) / (population_percentage / 100) = 28800 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l1043_104362


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l1043_104363

/-- Definition of a cuboid with given dimensions and surface area -/
structure Cuboid where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  surface_area : ℝ

/-- The surface area formula for a cuboid -/
def surface_area_formula (c : Cuboid) : ℝ :=
  2 * (c.edge1 * c.edge2 + c.edge1 * c.edge3 + c.edge2 * c.edge3)

/-- Theorem: For a cuboid with edges 4 cm, x cm, and 6 cm, and a surface area of 148 cm², 
    the length of the second edge (x) is 5 cm -/
theorem cuboid_edge_length (c : Cuboid) 
    (h1 : c.edge1 = 4)
    (h2 : c.edge3 = 6)
    (h3 : c.surface_area = 148)
    (h4 : surface_area_formula c = c.surface_area) :
    c.edge2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_cuboid_edge_length_l1043_104363


namespace NUMINAMATH_CALUDE_find_k_value_l1043_104344

/-- Given functions f and g, prove the value of k when f(3) - g(3) = 6 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^2 - 2 / x + 5) →
  (∀ x, g x = x^2 - k) →
  f 3 - g 3 = 6 →
  k = -157 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l1043_104344


namespace NUMINAMATH_CALUDE_stick_cutting_probability_l1043_104376

theorem stick_cutting_probability (stick_length : Real) (mark_position : Real) 
  (h1 : stick_length = 2)
  (h2 : mark_position = 0.6)
  (h3 : 0 < mark_position ∧ mark_position < stick_length) :
  let cut_range := stick_length - mark_position
  let valid_cut := min (stick_length / 4) cut_range
  (valid_cut / cut_range) = 5/14 := by
  sorry


end NUMINAMATH_CALUDE_stick_cutting_probability_l1043_104376


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l1043_104397

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Theorem stating that if Car O travels at three times the speed of Car P for the same distance,
    then Car O's travel time is one-third of Car P's travel time -/
theorem car_speed_time_relation (p o : Car) (distance : ℝ) :
  o.speed = 3 * p.speed →
  distance = p.speed * p.time →
  distance = o.speed * o.time →
  o.time = p.time / 3 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_time_relation_l1043_104397


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l1043_104337

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x ≤ -3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < 0 ∨ x ≥ 1} := by sorry

-- Theorem for the complement of A with respect to ℝ
theorem complement_A : Aᶜ = {x | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l1043_104337


namespace NUMINAMATH_CALUDE_abs_over_a_plus_one_l1043_104382

theorem abs_over_a_plus_one (a : ℝ) (h : a ≠ 0) :
  (|a| / a + 1 = 0) ∨ (|a| / a + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_abs_over_a_plus_one_l1043_104382


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1043_104312

theorem system_of_equations_solution (a b c d e f g : ℚ) : 
  a + b + c + d + e = 1 →
  b + c + d + e + f = 2 →
  c + d + e + f + g = 3 →
  d + e + f + g + a = 4 →
  e + f + g + a + b = 5 →
  f + g + a + b + c = 6 →
  g + a + b + c + d = 7 →
  g = 13/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1043_104312


namespace NUMINAMATH_CALUDE_pool_filling_cost_l1043_104300

/-- Proves that the cost to fill a pool is $5 given the specified conditions -/
theorem pool_filling_cost (fill_time : ℕ) (flow_rate : ℕ) (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate : ℚ) * water_cost = 5 := by
  sorry

#check pool_filling_cost

end NUMINAMATH_CALUDE_pool_filling_cost_l1043_104300


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1043_104316

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1043_104316


namespace NUMINAMATH_CALUDE_probability_of_three_in_six_sevenths_l1043_104388

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem probability_of_three_in_six_sevenths : 
  let rep := decimal_representation 6 7
  ∀ k, k ∈ rep → k ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_probability_of_three_in_six_sevenths_l1043_104388


namespace NUMINAMATH_CALUDE_unique_integer_l1043_104305

theorem unique_integer (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) : 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_l1043_104305


namespace NUMINAMATH_CALUDE_a2_times_a6_eq_68_l1043_104378

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℤ := 4 * n^2 - 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := S n - S (n-1)

/-- Theorem stating that a_2 * a_6 = 68 -/
theorem a2_times_a6_eq_68 : a 2 * a 6 = 68 := by
  sorry

end NUMINAMATH_CALUDE_a2_times_a6_eq_68_l1043_104378


namespace NUMINAMATH_CALUDE_science_team_selection_ways_l1043_104347

theorem science_team_selection_ways (total_boys : ℕ) (total_girls : ℕ) 
  (team_size : ℕ) (required_boys : ℕ) (required_girls : ℕ) : 
  total_boys = 7 → total_girls = 10 → team_size = 8 → 
  required_boys = 4 → required_girls = 4 →
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 7350 :=
by sorry

end NUMINAMATH_CALUDE_science_team_selection_ways_l1043_104347


namespace NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_is_four_l1043_104303

/-- The percentage of students who are freshmen psychology majors in the School of Liberal Arts -/
def freshmen_psych_liberal_arts_percentage (total_students : ℕ) : ℚ :=
  let freshmen_percentage : ℚ := 50 / 100
  let liberal_arts_percentage : ℚ := 40 / 100
  let psychology_percentage : ℚ := 20 / 100
  freshmen_percentage * liberal_arts_percentage * psychology_percentage * 100

theorem freshmen_psych_liberal_arts_percentage_is_four (total_students : ℕ) :
  freshmen_psych_liberal_arts_percentage total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_is_four_l1043_104303


namespace NUMINAMATH_CALUDE_initial_people_count_l1043_104384

/-- The number of people initially on the train -/
def initial_people : ℕ := sorry

/-- The number of people left on the train after the first stop -/
def people_left : ℕ := 31

/-- The number of people who got off at the first stop -/
def people_off : ℕ := 17

/-- Theorem stating that the initial number of people on the train was 48 -/
theorem initial_people_count : initial_people = people_left + people_off :=
by sorry

end NUMINAMATH_CALUDE_initial_people_count_l1043_104384


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1043_104389

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 + 5*x - 14) / (x^3 - 3*x^2 - x + 3) = 
    A / (x - 1) + B / (x - 3) + C / (x + 1) →
  A * B * C = -25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1043_104389


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1043_104322

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 192) 
  (h2 : weight_after = 66) : 
  weight_before - weight_after = 126 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1043_104322


namespace NUMINAMATH_CALUDE_rectangular_section_properties_l1043_104338

/-- A regular tetrahedron with unit edge length -/
structure UnitTetrahedron where
  -- Add necessary fields here

/-- A rectangular section of a tetrahedron -/
structure RectangularSection (T : UnitTetrahedron) where
  -- Add necessary fields here

/-- The perimeter of a rectangular section -/
def perimeter (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

/-- The area of a rectangular section -/
def area (T : UnitTetrahedron) (S : RectangularSection T) : ℝ :=
  sorry

theorem rectangular_section_properties (T : UnitTetrahedron) :
  (∀ S : RectangularSection T, perimeter T S = 2) ∧
  (∀ S : RectangularSection T, 0 ≤ area T S ∧ area T S ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_rectangular_section_properties_l1043_104338


namespace NUMINAMATH_CALUDE_car_rental_cost_l1043_104392

theorem car_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) :
  total_cost = 46.12 ∧ 
  miles_driven = 214 ∧ 
  cost_per_mile = 0.08 →
  ∃ daily_rental_cost : ℝ, 
    daily_rental_cost = 29 ∧ 
    total_cost = daily_rental_cost + miles_driven * cost_per_mile :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l1043_104392


namespace NUMINAMATH_CALUDE_point_m_coordinate_l1043_104311

/-- Given points L, M, N, P on a number line where M and N divide LP into three equal parts,
    prove that if L is at coordinate 1/6 and P is at coordinate 1/12, then M is at coordinate 1/9 -/
theorem point_m_coordinate (L M N P : ℝ) : 
  L = 1/6 →
  P = 1/12 →
  M - L = N - M →
  N - M = P - N →
  M = 1/9 := by sorry

end NUMINAMATH_CALUDE_point_m_coordinate_l1043_104311


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1043_104359

-- Define the sets P and Q
def P : Set ℝ := {x | |x| < 2}
def Q : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2) + 1}

-- State the theorem
theorem intersection_P_Q :
  P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1043_104359


namespace NUMINAMATH_CALUDE_inequality_proofs_l1043_104301

theorem inequality_proofs :
  (∀ x : ℝ, x * (1 - x) ≤ 1 / 4) ∧
  (∀ x a : ℝ, x * (a - x) ≤ a^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1043_104301


namespace NUMINAMATH_CALUDE_sequence_arrangement_count_l1043_104325

theorem sequence_arrangement_count : ℕ := by
  -- Define the length of the sequence
  let n : ℕ := 6

  -- Define the counts of each number in the sequence
  let count_of_ones : ℕ := 3
  let count_of_twos : ℕ := 2
  let count_of_threes : ℕ := 1

  -- Assert that the sum of counts equals the sequence length
  have h_sum_counts : count_of_ones + count_of_twos + count_of_threes = n := by sorry

  -- Define the number of ways to arrange the sequence
  let arrangement_count : ℕ := n.choose count_of_threes * (n - count_of_threes).choose count_of_twos

  -- Prove that the arrangement count equals 60
  have h_arrangement_count : arrangement_count = 60 := by sorry

  -- Return the final result
  exact 60

end NUMINAMATH_CALUDE_sequence_arrangement_count_l1043_104325


namespace NUMINAMATH_CALUDE_flowers_given_to_brother_correct_flowers_given_l1043_104320

theorem flowers_given_to_brother (amanda_flowers : ℕ) (peter_flowers_left : ℕ) : ℕ :=
  let peter_initial_flowers := 3 * amanda_flowers
  peter_initial_flowers - peter_flowers_left

theorem correct_flowers_given (amanda_flowers : ℕ) (peter_flowers_left : ℕ)
    (h1 : amanda_flowers = 20)
    (h2 : peter_flowers_left = 45) :
    flowers_given_to_brother amanda_flowers peter_flowers_left = 15 := by
  sorry

end NUMINAMATH_CALUDE_flowers_given_to_brother_correct_flowers_given_l1043_104320


namespace NUMINAMATH_CALUDE_competition_participants_l1043_104398

theorem competition_participants (freshmen : ℕ) (sophomores : ℕ) : 
  freshmen = 8 → sophomores = 5 * freshmen → freshmen + sophomores = 48 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l1043_104398


namespace NUMINAMATH_CALUDE_electricity_pricing_l1043_104395

/-- Represents the electricity pricing problem -/
theorem electricity_pricing
  (a : ℝ) -- annual electricity consumption in kilowatt-hours
  (x : ℝ) -- new electricity price per kilowatt-hour
  (h1 : 0 < a) -- assumption that consumption is positive
  (h2 : 0.55 ≤ x ∧ x ≤ 0.75) -- new price range
  : ((0.2 * a / (x - 0.40) + a) * (x - 0.30) ≥ 0.60 * a) ↔ (x ≥ 0.60) :=
by sorry

end NUMINAMATH_CALUDE_electricity_pricing_l1043_104395


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l1043_104327

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem second_term_of_sequence (a d : ℤ) :
  arithmetic_sequence a d 12 = 11 →
  arithmetic_sequence a d 13 = 14 →
  arithmetic_sequence a d 2 = -19 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l1043_104327


namespace NUMINAMATH_CALUDE_larger_number_proof_l1043_104317

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1043_104317


namespace NUMINAMATH_CALUDE_grid_drawing_theorem_l1043_104387

/-- Represents a grid configuration -/
structure GridConfiguration (n : ℕ+) :=
  (has_diagonal : Fin n → Fin n → Bool)
  (start_vertex : Fin n × Fin n)
  (is_valid : Bool)

/-- Checks if a grid configuration is valid according to the problem conditions -/
def is_valid_configuration (n : ℕ+) (config : GridConfiguration n) : Prop :=
  -- Adjacent cells have diagonals in different directions
  (∀ i j, config.has_diagonal i j → ¬(config.has_diagonal (i+1) j ∧ config.has_diagonal i (j+1))) ∧
  -- Can be drawn in one stroke starting from bottom-left vertex
  (config.start_vertex = (0, 0)) ∧
  -- Each edge or diagonal is traversed exactly once
  config.is_valid

/-- The main theorem stating that only n = 1, 2, 3 satisfy the conditions -/
theorem grid_drawing_theorem :
  ∀ n : ℕ+, (∃ config : GridConfiguration n, is_valid_configuration n config) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_grid_drawing_theorem_l1043_104387


namespace NUMINAMATH_CALUDE_gardening_club_membership_l1043_104368

theorem gardening_club_membership (initial_total : ℕ) 
  (h1 : initial_total > 0)
  (h2 : (60 : ℚ) / 100 * initial_total = (initial_total * 3) / 5) 
  (h3 : (((initial_total * 3) / 5 - 3 : ℚ) / initial_total) = 1 / 2) : 
  (initial_total * 3) / 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_gardening_club_membership_l1043_104368


namespace NUMINAMATH_CALUDE_emily_necklaces_l1043_104321

def beads_per_necklace : ℕ := 28
def total_beads : ℕ := 308

theorem emily_necklaces :
  (total_beads / beads_per_necklace : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_emily_necklaces_l1043_104321


namespace NUMINAMATH_CALUDE_smallest_odd_n_l1043_104365

def is_smallest_odd (n : ℕ) : Prop :=
  Odd n ∧ 
  (3 : ℝ) ^ ((n + 1)^2 / 5) > 500 ∧ 
  ∀ m : ℕ, Odd m ∧ m < n → (3 : ℝ) ^ ((m + 1)^2 / 5) ≤ 500

theorem smallest_odd_n : is_smallest_odd 6 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_n_l1043_104365


namespace NUMINAMATH_CALUDE_sum_a_b_equals_seven_l1043_104383

/-- Represents a four-digit number in the form 3a72 -/
def fourDigitNum (a : ℕ) : ℕ := 3000 + 100 * a + 72

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

theorem sum_a_b_equals_seven :
  ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (fourDigitNum a + 895 = 4000 + 100 * b + 67) →
  divisibleBy11 (4000 + 100 * b + 67) →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_seven_l1043_104383


namespace NUMINAMATH_CALUDE_total_minutes_worked_l1043_104332

/-- Calculates the total minutes worked by three people given specific conditions -/
theorem total_minutes_worked (bianca_hours : ℝ) : 
  bianca_hours = 12.5 → 
  (3 * bianca_hours + bianca_hours - 8.5) * 60 = 3240 := by
  sorry

#check total_minutes_worked

end NUMINAMATH_CALUDE_total_minutes_worked_l1043_104332


namespace NUMINAMATH_CALUDE_qingming_rain_is_random_l1043_104346

/-- An event that occurs during a specific season --/
structure SeasonalEvent where
  season : String
  description : String

/-- A property indicating whether an event can be predicted with certainty --/
def isPredictable (e : SeasonalEvent) : Prop := sorry

/-- A property indicating whether an event's occurrence varies from year to year --/
def hasVariableOccurrence (e : SeasonalEvent) : Prop := sorry

/-- Definition of a random event --/
def isRandomEvent (e : SeasonalEvent) : Prop := 
  ¬(isPredictable e) ∧ hasVariableOccurrence e

/-- The main theorem --/
theorem qingming_rain_is_random (e : SeasonalEvent) 
  (h1 : e.season = "Qingming")
  (h2 : e.description = "drizzling rain")
  (h3 : ¬(isPredictable e))
  (h4 : hasVariableOccurrence e) : 
  isRandomEvent e := by
  sorry

end NUMINAMATH_CALUDE_qingming_rain_is_random_l1043_104346


namespace NUMINAMATH_CALUDE_quadratic_discriminant_value_l1043_104304

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant_value (a : ℝ) :
  discriminant 1 (-3) (-2*a) = 1 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_value_l1043_104304


namespace NUMINAMATH_CALUDE_range_of_m_l1043_104330

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- Theorem statement
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1043_104330


namespace NUMINAMATH_CALUDE_part_one_part_two_l1043_104336

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x ≥ 1} = Set.Icc 0 (4/3) := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x - |2*x - 5| ≤ 0) →
  a ∈ Set.Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1043_104336


namespace NUMINAMATH_CALUDE_inequality_implication_l1043_104350

theorem inequality_implication (x y : ℝ) (h : x > y) : 2*x - 1 > 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1043_104350


namespace NUMINAMATH_CALUDE_landscape_length_l1043_104315

/-- Given a rectangular landscape with a playground, calculate its length -/
theorem landscape_length (breadth : ℝ) (playground_area : ℝ) : 
  breadth > 0 →
  playground_area = 1200 →
  playground_area = (1 / 6) * (8 * breadth * breadth) →
  8 * breadth = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l1043_104315


namespace NUMINAMATH_CALUDE_union_complement_problem_l1043_104356

theorem union_complement_problem (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {3, 4}) (hB : B = {1, 4, 5}) :
  A ∪ (U \ B) = {2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1043_104356


namespace NUMINAMATH_CALUDE_officers_selection_count_l1043_104367

/-- The number of ways to choose officers from a club -/
def choose_officers (total_members : ℕ) (senior_members : ℕ) (positions : ℕ) : ℕ :=
  senior_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem officers_selection_count :
  choose_officers 12 4 5 = 31680 := by
  sorry

end NUMINAMATH_CALUDE_officers_selection_count_l1043_104367


namespace NUMINAMATH_CALUDE_diameter_endpoints_form_trapezoid_l1043_104364

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) : Prop :=
  -- Circles are external to each other
  let (x₁, y₁) := c₁.center
  let (x₂, y₂) := c₂.center
  (x₁ - x₂)^2 + (y₁ - y₂)^2 > (c₁.radius + c₂.radius)^2 ∧
  -- d₁ and d₂ are diameters of c₁ and c₂ respectively
  (∀ p ∈ d₁, dist p c₁.center ≤ c₁.radius) ∧
  (∀ p ∈ d₂, dist p c₂.center ≤ c₂.radius) ∧
  -- The line through one diameter is tangent to the other circle
  (∃ p ∈ d₁, dist p c₂.center = c₂.radius) ∧
  (∃ p ∈ d₂, dist p c₁.center = c₁.radius)

-- Define a trapezoid
def is_trapezoid (quadrilateral : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ), quadrilateral = {a, b, c, d} ∧
  (∃ (m : ℝ), (c.1 - d.1 = m * (a.1 - b.1) ∧ c.2 - d.2 = m * (a.2 - b.2)) ∨
              (b.1 - c.1 = m * (a.1 - d.1) ∧ b.2 - c.2 = m * (a.2 - d.2)))

-- Theorem statement
theorem diameter_endpoints_form_trapezoid (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) :
  problem_setup c₁ c₂ d₁ d₂ →
  is_trapezoid (d₁ ∪ d₂) :=
sorry

end NUMINAMATH_CALUDE_diameter_endpoints_form_trapezoid_l1043_104364


namespace NUMINAMATH_CALUDE_subtraction_problem_l1043_104314

theorem subtraction_problem (M N : ℕ) : 
  M < 10 → N < 10 → M * 10 + 4 - (30 + N) = 16 → M + N = 13 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1043_104314


namespace NUMINAMATH_CALUDE_product_equality_l1043_104307

theorem product_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l1043_104307


namespace NUMINAMATH_CALUDE_range_of_a_l1043_104331

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * Real.exp (a * Real.exp x + a) ≥ Real.log (Real.exp x + 1)) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1043_104331


namespace NUMINAMATH_CALUDE_regular_tetrahedron_iff_l1043_104324

/-- A tetrahedron -/
structure Tetrahedron where
  /-- The base of the tetrahedron -/
  base : Triangle
  /-- The apex of the tetrahedron -/
  apex : Point

/-- A regular tetrahedron -/
def RegularTetrahedron (t : Tetrahedron) : Prop :=
  sorry

/-- The base of the tetrahedron is an equilateral triangle -/
def HasEquilateralBase (t : Tetrahedron) : Prop :=
  sorry

/-- The dihedral angles between the lateral faces and the base are equal -/
def HasEqualDihedralAngles (t : Tetrahedron) : Prop :=
  sorry

/-- All lateral edges form equal angles with the base -/
def HasEqualLateralEdgeAngles (t : Tetrahedron) : Prop :=
  sorry

/-- Theorem: A tetrahedron is regular if and only if it satisfies certain conditions -/
theorem regular_tetrahedron_iff (t : Tetrahedron) : 
  RegularTetrahedron t ↔ 
  (HasEquilateralBase t ∧ HasEqualDihedralAngles t) ∨
  (HasEqualLateralEdgeAngles t ∧ HasEqualDihedralAngles t) :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_iff_l1043_104324


namespace NUMINAMATH_CALUDE_homework_assignment_question_distribution_l1043_104353

theorem homework_assignment_question_distribution :
  ∃! (x y z : ℕ),
    x + y + z = 100 ∧
    (0.5 : ℝ) * x + 3 * y + 10 * z = 100 ∧
    x = 80 ∧ y = 20 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_homework_assignment_question_distribution_l1043_104353


namespace NUMINAMATH_CALUDE_custom_mul_solution_l1043_104351

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that if a * 4 = 9 under the custom multiplication, then a = 12.5 -/
theorem custom_mul_solution :
  ∃ a : ℝ, custom_mul a 4 = 9 ∧ a = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l1043_104351


namespace NUMINAMATH_CALUDE_lotto_game_minimum_draws_l1043_104390

theorem lotto_game_minimum_draws (n : ℕ) (h : n = 90) : 
  ∃ k : ℕ, k = 49 ∧ 
  (∀ S : Finset ℕ, S.card = k → S ⊆ Finset.range n → 
    ∃ x ∈ S, x % 3 = 0 ∨ x % 5 = 0) ∧
  (∀ m : ℕ, m < k → 
    ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range n ∧ 
    ∀ x ∈ T, x % 3 ≠ 0 ∧ x % 5 ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_lotto_game_minimum_draws_l1043_104390


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1043_104381

-- Define a 3x3 grid as a function from (Fin 3 × Fin 3) to ℕ
def Grid := Fin 3 → Fin 3 → ℕ

-- Define a predicate to check if a number is between 1 and 9
def InRange (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define a predicate to check if a grid contains all numbers from 1 to 9
def ContainsAllNumbers (g : Grid) : Prop :=
  ∀ n, InRange n → ∃ i j, g i j = n

-- Define a predicate to check if the product of numbers in a row is a multiple of 4
def RowProductMultipleOf4 (g : Grid) : Prop :=
  ∀ i, (g i 0) * (g i 1) * (g i 2) % 4 = 0

-- Define a predicate to check if the product of numbers in a column is a multiple of 4
def ColProductMultipleOf4 (g : Grid) : Prop :=
  ∀ j, (g 0 j) * (g 1 j) * (g 2 j) % 4 = 0

-- The main theorem
theorem impossible_arrangement : ¬∃ (g : Grid),
  ContainsAllNumbers g ∧ 
  RowProductMultipleOf4 g ∧ 
  ColProductMultipleOf4 g :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1043_104381


namespace NUMINAMATH_CALUDE_square_sum_equals_43_l1043_104326

theorem square_sum_equals_43 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_43_l1043_104326


namespace NUMINAMATH_CALUDE_sqrt_a_squared_b_l1043_104318

theorem sqrt_a_squared_b (a b : ℝ) (h : a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_b_l1043_104318


namespace NUMINAMATH_CALUDE_sequence_count_16_l1043_104335

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 2 => 2 * validSequences n

/-- The problem statement -/
theorem sequence_count_16 : validSequences 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_sequence_count_16_l1043_104335


namespace NUMINAMATH_CALUDE_range_of_m_l1043_104341

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  4 * x₁^2 - 4*(m-2)*x₁ + 1 = 0 ∧ 4 * x₂^2 - 4*(m-2)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 3*m*x + 1 ≠ 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) →
  (m ≤ -2/3) ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1043_104341


namespace NUMINAMATH_CALUDE_total_shaded_area_is_107_l1043_104343

/-- Represents a rectangle in a plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle in a plane -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

/-- Represents the configuration of shapes in the plane -/
structure ShapeConfiguration where
  rect1 : Rectangle
  rect2 : Rectangle
  triangle : Triangle
  rect1TriangleOverlap : ℝ
  rect2TriangleOverlap : ℝ
  rectOverlap : ℝ

/-- Calculates the total shaded area given a ShapeConfiguration -/
def totalShadedArea (config : ShapeConfiguration) : ℝ :=
  rectangleArea config.rect1 + rectangleArea config.rect2 + triangleArea config.triangle -
  config.rectOverlap - config.rect1TriangleOverlap - config.rect2TriangleOverlap

/-- The theorem stating the total shaded area for the given configuration -/
theorem total_shaded_area_is_107 (config : ShapeConfiguration) :
  config.rect1 = ⟨5, 12⟩ →
  config.rect2 = ⟨4, 15⟩ →
  config.triangle = ⟨3, 4⟩ →
  config.rect1TriangleOverlap = 2 →
  config.rect2TriangleOverlap = 1 →
  config.rectOverlap = 16 →
  totalShadedArea config = 107 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_107_l1043_104343


namespace NUMINAMATH_CALUDE_sum_of_integers_l1043_104349

theorem sum_of_integers (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x = y + 3) (h4 : x^3 - y^3 = 63) :
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1043_104349


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1043_104361

theorem complex_equation_solution (z : ℂ) : 
  z^2 + 2*Complex.I*z + 3 = 0 ↔ z = Complex.I ∨ z = -3*Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1043_104361


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1043_104366

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1043_104366


namespace NUMINAMATH_CALUDE_trig_identity_l1043_104339

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1043_104339


namespace NUMINAMATH_CALUDE_base_r_problem_l1043_104342

/-- Represents a number in base r -/
def BaseRNumber (r : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem base_r_problem (r : ℕ) : 
  (r > 1) →
  (BaseRNumber r [0, 0, 0, 1] = 1000) →
  (BaseRNumber r [0, 4, 4] = 440) →
  (BaseRNumber r [0, 4, 3] = 340) →
  (1000 - 440 = 340) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_base_r_problem_l1043_104342


namespace NUMINAMATH_CALUDE_complex_below_real_axis_l1043_104310

theorem complex_below_real_axis (t : ℝ) : 
  let z : ℂ := (2 * t^2 + 5 * t - 3) + (t^2 + 2 * t + 2) * I
  Complex.im z < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_below_real_axis_l1043_104310


namespace NUMINAMATH_CALUDE_min_segments_on_cube_edges_l1043_104373

/-- A cube representation -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- A broken line on the surface of a cube -/
structure BrokenLine where
  segments : Finset (Fin 8 × Fin 8)
  num_segments : Nat
  is_closed : Bool
  vertices_on_cube : Bool

/-- Theorem statement -/
theorem min_segments_on_cube_edges (c : Cube) (bl : BrokenLine) :
  bl.num_segments = 8 ∧ bl.is_closed ∧ bl.vertices_on_cube →
  ∃ (coinciding_segments : Finset (Fin 8 × Fin 8)),
    coinciding_segments ⊆ c.edges ∧
    coinciding_segments ⊆ bl.segments ∧
    coinciding_segments.card = 2 ∧
    ∀ (cs : Finset (Fin 8 × Fin 8)),
      cs ⊆ c.edges ∧ cs ⊆ bl.segments →
      cs.card ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_segments_on_cube_edges_l1043_104373


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l1043_104375

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The factorial of 8 -/
def blood_expiration_seconds : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- The day a unit of blood expires when donated at noon -/
def blood_expiration_day (donation_day : ℕ) : ℕ :=
  donation_day + (blood_expiration_seconds / seconds_per_day)

theorem blood_expires_same_day (donation_day : ℕ) :
  blood_expiration_day donation_day = donation_day := by
  sorry

end NUMINAMATH_CALUDE_blood_expires_same_day_l1043_104375


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1043_104333

theorem arithmetic_geometric_mean_ratio (a b m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hm : (a + b) / 2 = m * Real.sqrt (a * b)) : 
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1043_104333


namespace NUMINAMATH_CALUDE_at_most_three_lines_unique_line_through_two_points_l1043_104306

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to create a line from two points
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

-- Theorem 1: At most three lines can be drawn through any two of three distinct points
theorem at_most_three_lines (p1 p2 p3 : Point) 
  (h1 : p1 ≠ p2) (h2 : p2 ≠ p3) (h3 : p1 ≠ p3) : 
  ∃ (l1 l2 l3 : Line), ∀ (l : Line), 
    (isPointOnLine p1 l ∧ isPointOnLine p2 l) ∨
    (isPointOnLine p2 l ∧ isPointOnLine p3 l) ∨
    (isPointOnLine p1 l ∧ isPointOnLine p3 l) →
    l = l1 ∨ l = l2 ∨ l = l3 :=
sorry

-- Theorem 2: Only one line can be drawn through two distinct points
theorem unique_line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! (l : Line), isPointOnLine p1 l ∧ isPointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_at_most_three_lines_unique_line_through_two_points_l1043_104306


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1043_104379

theorem repeating_decimal_sum (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (3 : ℚ) / 13 = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + 
    (a : ℚ) / 1000 + (b : ℚ) / 10000 + 
    (a : ℚ) / 100000 + (b : ℚ) / 1000000 + 
    (a : ℚ) / 10000000 + (b : ℚ) / 100000000 + 
    (a : ℚ) / 1000000000 + (b : ℚ) / 10000000000 →
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1043_104379


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_inches_l1043_104302

/-- Converts cubic yards to cubic inches -/
def cubic_yards_to_cubic_inches (yards : ℕ) : ℕ :=
  let feet_per_yard : ℕ := 3
  let inches_per_foot : ℕ := 12
  yards * (feet_per_yard ^ 3) * (inches_per_foot ^ 3)

/-- Theorem stating that 5 cubic yards equals 233280 cubic inches -/
theorem five_cubic_yards_to_cubic_inches :
  cubic_yards_to_cubic_inches 5 = 233280 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_inches_l1043_104302


namespace NUMINAMATH_CALUDE_polynomial_inequality_polynomial_inequality_equality_condition_l1043_104377

/-- A polynomial of degree 3 with roots in (0, 1) -/
structure PolynomialWithRootsInUnitInterval where
  b : ℝ
  c : ℝ
  root_property : ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < 1 ∧
                                    0 < x₂ ∧ x₂ < 1 ∧
                                    0 < x₃ ∧ x₃ < 1 ∧
                                    x₁ + x₂ + x₃ = 2 ∧
                                    x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = b ∧
                                    x₁ * x₂ * x₃ = -c

/-- The main theorem stating the inequality for polynomials with roots in (0, 1) -/
theorem polynomial_inequality (P : PolynomialWithRootsInUnitInterval) :
  8 * P.b + 9 * P.c ≤ 8 := by
  sorry

/-- Conditions for equality in the polynomial inequality -/
theorem polynomial_inequality_equality_condition (P : PolynomialWithRootsInUnitInterval) :
  (8 * P.b + 9 * P.c = 8) ↔ 
  (∃ (x : ℝ), x = 2/3 ∧ 
   ∃ (x₁ x₂ x₃ : ℝ), x₁ = x ∧ x₂ = x ∧ x₃ = x ∧
                     0 < x₁ ∧ x₁ < 1 ∧
                     0 < x₂ ∧ x₂ < 1 ∧
                     0 < x₃ ∧ x₃ < 1 ∧
                     x₁ + x₂ + x₃ = 2 ∧
                     x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = P.b ∧
                     x₁ * x₂ * x₃ = -P.c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_polynomial_inequality_equality_condition_l1043_104377


namespace NUMINAMATH_CALUDE_complex_power_2017_l1043_104354

theorem complex_power_2017 : ((1 - Complex.I) / (1 + Complex.I)) ^ 2017 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2017_l1043_104354


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1043_104348

theorem complex_expression_evaluation : 
  let z₁ : ℂ := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z₂ : ℂ := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z₃ : ℂ := 1 / (8 * Complex.I^3)
  z₁ + z₂ + z₃ = -1.6 + 0.125 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1043_104348


namespace NUMINAMATH_CALUDE_geometric_sum_6_terms_l1043_104371

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 2 and common ratio 1/3 -/
theorem geometric_sum_6_terms :
  geometricSum 2 (1/3) 6 = 728/243 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_6_terms_l1043_104371


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_50_l1043_104374

theorem units_digit_of_7_to_50 : 7^50 ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_50_l1043_104374


namespace NUMINAMATH_CALUDE_box_sum_equals_sixteen_l1043_104355

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) + (b ^ c : ℚ) - (c ^ a : ℚ)

theorem box_sum_equals_sixteen : box 2 3 (-1) + box (-1) 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_sum_equals_sixteen_l1043_104355


namespace NUMINAMATH_CALUDE_EF_length_is_19_2_l1043_104369

/-- Two similar right triangles ABC and DEF with given side lengths -/
structure SimilarRightTriangles where
  -- Triangle ABC
  AB : ℝ
  BC : ℝ
  -- Triangle DEF
  DE : ℝ
  -- Similarity ratio
  k : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  DE_positive : DE > 0
  k_positive : k > 0
  similarity : k = DE / AB
  AB_value : AB = 10
  BC_value : BC = 8
  DE_value : DE = 24

/-- The length of EF in the similar right triangles -/
def EF_length (t : SimilarRightTriangles) : ℝ :=
  t.k * t.BC

/-- Theorem: The length of EF is 19.2 -/
theorem EF_length_is_19_2 (t : SimilarRightTriangles) : EF_length t = 19.2 := by
  sorry

#check EF_length_is_19_2

end NUMINAMATH_CALUDE_EF_length_is_19_2_l1043_104369


namespace NUMINAMATH_CALUDE_largest_pile_size_l1043_104380

theorem largest_pile_size (total : ℕ) (small medium large : ℕ) : 
  total = small + medium + large →
  medium = 2 * small →
  large = 3 * small →
  total = 240 →
  large = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_pile_size_l1043_104380


namespace NUMINAMATH_CALUDE_cloth_sale_cost_price_l1043_104329

/-- Given the conditions of a cloth sale, prove the cost price per metre -/
theorem cloth_sale_cost_price 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (discount_rate : ℚ) 
  (tax_rate : ℚ) 
  (h1 : total_metres = 300)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5)
  (h4 : discount_rate = 1/10)
  (h5 : tax_rate = 1/20)
  : ℕ := by
  sorry

#check cloth_sale_cost_price

end NUMINAMATH_CALUDE_cloth_sale_cost_price_l1043_104329


namespace NUMINAMATH_CALUDE_cookies_leftover_l1043_104328

/-- The number of cookies Amelia has -/
def ameliaCookies : ℕ := 52

/-- The number of cookies Benjamin has -/
def benjaminCookies : ℕ := 63

/-- The number of cookies Chloe has -/
def chloeCookies : ℕ := 25

/-- The number of cookies in each package -/
def packageSize : ℕ := 15

/-- The total number of cookies -/
def totalCookies : ℕ := ameliaCookies + benjaminCookies + chloeCookies

/-- The number of cookies left over after packaging -/
def leftoverCookies : ℕ := totalCookies % packageSize

theorem cookies_leftover : leftoverCookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_l1043_104328


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l1043_104309

theorem train_platform_passing_time :
  let train_length : ℝ := 360
  let platform_length : ℝ := 390
  let train_speed_kmh : ℝ := 45
  let total_distance : ℝ := train_length + platform_length
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let time : ℝ := total_distance / train_speed_ms
  time = 60 := by sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l1043_104309


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1043_104399

theorem hyperbola_eccentricity (a b e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt (2 * e - 1) * x) →
  e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1043_104399


namespace NUMINAMATH_CALUDE_first_group_count_l1043_104372

theorem first_group_count (total_count : Nat) (total_avg : ℝ) (first_group_avg : ℝ) 
  (last_group_count : Nat) (last_group_avg : ℝ) (sixth_number : ℝ) : 
  total_count = 11 →
  total_avg = 10.7 →
  first_group_avg = 10.5 →
  last_group_count = 6 →
  last_group_avg = 11.4 →
  sixth_number = 13.700000000000017 →
  (total_count - last_group_count : ℝ) = 4 := by
sorry

end NUMINAMATH_CALUDE_first_group_count_l1043_104372


namespace NUMINAMATH_CALUDE_university_theater_sales_l1043_104308

/-- The total money made from ticket sales at University Theater --/
def total_money_made (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - senior_tickets
  adult_tickets * adult_price + senior_tickets * senior_price

/-- Theorem: The University Theater made $8748 from ticket sales --/
theorem university_theater_sales : total_money_made 510 21 15 327 = 8748 := by
  sorry

end NUMINAMATH_CALUDE_university_theater_sales_l1043_104308


namespace NUMINAMATH_CALUDE_jason_money_calculation_l1043_104360

/-- Represents the value of coins in cents -/
inductive Coin
  | quarter
  | dime
  | nickel

/-- The value of a coin in cents -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.quarter => 25
  | Coin.dime => 10
  | Coin.nickel => 5

/-- Calculates the total value of coins in dollars -/
def coins_value (quarters dimes nickels : ℕ) : ℚ :=
  (quarters * coin_value Coin.quarter + dimes * coin_value Coin.dime + nickels * coin_value Coin.nickel) / 100

/-- Converts euros to US dollars -/
def euros_to_dollars (euros : ℚ) : ℚ :=
  euros * 1.20

theorem jason_money_calculation (initial_quarters initial_dimes initial_nickels : ℕ)
    (initial_euros : ℚ)
    (additional_quarters additional_dimes additional_nickels : ℕ)
    (additional_euros : ℚ) :
    let initial_coins := coins_value initial_quarters initial_dimes initial_nickels
    let initial_dollars := initial_coins + euros_to_dollars initial_euros
    let additional_coins := coins_value additional_quarters additional_dimes additional_nickels
    let additional_dollars := additional_coins + euros_to_dollars additional_euros
    let total_dollars := initial_dollars + additional_dollars
    initial_quarters = 49 →
    initial_dimes = 32 →
    initial_nickels = 18 →
    initial_euros = 22.50 →
    additional_quarters = 25 →
    additional_dimes = 15 →
    additional_nickels = 10 →
    additional_euros = 12 →
    total_dollars = 66 := by
  sorry

end NUMINAMATH_CALUDE_jason_money_calculation_l1043_104360


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1043_104340

universe u

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1043_104340
