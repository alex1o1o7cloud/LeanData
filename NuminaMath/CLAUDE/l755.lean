import Mathlib

namespace NUMINAMATH_CALUDE_product_abcd_zero_l755_75514

theorem product_abcd_zero 
  (a b c d : ℝ) 
  (eq1 : 3*a + 2*b + 4*c + 6*d = 60)
  (eq2 : 4*(d+c) = b^2)
  (eq3 : 4*b + 2*c = a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_zero_l755_75514


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l755_75507

theorem smallest_n_for_unique_k : ∃ (k : ℤ), (9 : ℚ)/17 < (3 : ℚ)/(3 + k) ∧ (3 : ℚ)/(3 + k) < 8/15 ∧
  ∀ (n : ℕ), n < 3 → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l755_75507


namespace NUMINAMATH_CALUDE_initial_average_problem_l755_75526

theorem initial_average_problem (initial_count : Nat) (new_value : ℝ) (average_decrease : ℝ) :
  initial_count = 6 →
  new_value = 7 →
  average_decrease = 1 →
  ∃ initial_average : ℝ,
    initial_average * initial_count + new_value = 
    (initial_average - average_decrease) * (initial_count + 1) ∧
    initial_average = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_problem_l755_75526


namespace NUMINAMATH_CALUDE_geometric_series_equality_l755_75508

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℝ := λ k => 512 * (1 - (1 / 2^k))
  let D : ℕ → ℝ := λ k => (2048 / 3) * (1 - (-1 / 2)^k)
  (∀ k < n, C k ≠ D k) ∧ C n = D n → n = 4
) := by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l755_75508


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l755_75510

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 3 * a 7 = 8
  property2 : a 4 + a 6 = 6

/-- Theorem: For a geometric sequence satisfying the given properties, a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 2 + seq.a 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l755_75510


namespace NUMINAMATH_CALUDE_used_car_selection_l755_75597

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 10 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l755_75597


namespace NUMINAMATH_CALUDE_remainder_proof_l755_75524

theorem remainder_proof :
  let n : ℕ := 174
  let d₁ : ℕ := 34
  let d₂ : ℕ := 5
  (n % d₁ = 4) ∧ (n % d₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_proof_l755_75524


namespace NUMINAMATH_CALUDE_seashells_left_l755_75502

def total_seashells : ℕ := 679
def clam_shells : ℕ := 325
def conch_shells : ℕ := 210
def oyster_shells : ℕ := 144
def starfish : ℕ := 110

def clam_percentage : ℚ := 40 / 100
def conch_percentage : ℚ := 25 / 100
def oyster_fraction : ℚ := 1 / 3

theorem seashells_left : 
  (clam_shells - Int.floor (clam_percentage * clam_shells)) +
  (conch_shells - Int.ceil (conch_percentage * conch_shells)) +
  (oyster_shells - Int.floor (oyster_fraction * oyster_shells)) +
  starfish = 558 := by
sorry

end NUMINAMATH_CALUDE_seashells_left_l755_75502


namespace NUMINAMATH_CALUDE_largest_multiple_13_negation_gt_neg150_l755_75518

theorem largest_multiple_13_negation_gt_neg150 : 
  ∀ n : ℤ, n * 13 > 143 → -(n * 13) ≤ -150 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_13_negation_gt_neg150_l755_75518


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l755_75595

/-- Given a line segment from (2,2) to (x,6) with length 5 and x > 0, prove x = 5 -/
theorem line_segment_endpoint (x : ℝ) 
  (h1 : (x - 2)^2 + (6 - 2)^2 = 5^2) 
  (h2 : x > 0) : 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l755_75595


namespace NUMINAMATH_CALUDE_triangle_properties_l755_75560

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) (R : Real) :
  2 * Real.sqrt 3 * (Real.sin (abc.A / 2))^2 + Real.sin abc.A - Real.sqrt 3 = 0 →
  (1/2) * abc.b * abc.c * Real.sin abc.A = Real.sqrt 3 →
  R = Real.sqrt 3 →
  abc.A = π/3 ∧ abc.a + abc.b + abc.c = 3 + Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l755_75560


namespace NUMINAMATH_CALUDE_special_linear_function_unique_l755_75565

/-- A linear function f such that f(f(x)) = x + 2 -/
def special_linear_function (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = x + 2)

/-- The unique linear function satisfying f(f(x)) = x + 2 is f(x) = x + 1 -/
theorem special_linear_function_unique (f : ℝ → ℝ) :
  special_linear_function f → (∀ x, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_special_linear_function_unique_l755_75565


namespace NUMINAMATH_CALUDE_dishes_for_equal_time_l755_75532

/-- Represents the time taken for different chores -/
structure ChoreTime where
  sweep : ℕ  -- minutes per room
  wash : ℕ   -- minutes per dish
  laundry : ℕ -- minutes per load
  dust : ℕ   -- minutes per surface

/-- Represents the chores assigned to Anna -/
structure AnnaChores where
  rooms : ℕ
  surfaces : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  loads : ℕ
  surfaces : ℕ

/-- Calculates the total time Anna spends on chores -/
def annaTime (ct : ChoreTime) (ac : AnnaChores) : ℕ :=
  ct.sweep * ac.rooms + ct.dust * ac.surfaces

/-- Calculates the total time Billy spends on chores, excluding dishes -/
def billyTimeBeforeDishes (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  ct.laundry * bc.loads + ct.dust * bc.surfaces

/-- The main theorem to prove -/
theorem dishes_for_equal_time (ct : ChoreTime) (ac : AnnaChores) (bc : BillyChores) :
  ct.sweep = 3 →
  ct.wash = 2 →
  ct.laundry = 9 →
  ct.dust = 1 →
  ac.rooms = 10 →
  ac.surfaces = 14 →
  bc.loads = 2 →
  bc.surfaces = 6 →
  ∃ (dishes : ℕ), dishes = 10 ∧
    annaTime ct ac = billyTimeBeforeDishes ct bc + ct.wash * dishes :=
by
  sorry


end NUMINAMATH_CALUDE_dishes_for_equal_time_l755_75532


namespace NUMINAMATH_CALUDE_unique_solution_when_a_is_three_fourths_l755_75579

/-- The equation has exactly one solution when a = 3/4 -/
theorem unique_solution_when_a_is_three_fourths (x a : ℝ) :
  (∃! x, (x^2 - a)^2 + 2*(x^2 - a) + (x - a) + 2 = 0) ↔ a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_a_is_three_fourths_l755_75579


namespace NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l755_75563

/-- Represents a player in the card game -/
inductive Player : Type
| one : Player
| two : Player
| three : Player
| four : Player

/-- The total number of cards in the deck -/
def deck_size : ℕ := 32

/-- The number of players in the game -/
def num_players : ℕ := 4

/-- The number of Aces in the deck -/
def num_aces : ℕ := 4

/-- Calculates the probability of a player getting the first Ace -/
def prob_first_ace (p : Player) : ℚ :=
  1 / (deck_size / num_players)

/-- Theorem stating that the probability of each player getting the first Ace is 1/8 -/
theorem prob_first_ace_is_one_eighth (p : Player) :
  prob_first_ace p = 1 / 8 := by
  sorry

#check prob_first_ace_is_one_eighth

end NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l755_75563


namespace NUMINAMATH_CALUDE_terminal_side_half_angle_l755_75567

def is_in_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem terminal_side_half_angle (α : Real) :
  is_in_first_quadrant α → is_in_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_half_angle_l755_75567


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l755_75591

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem tenth_term_of_sequence :
  let a₁ : ℤ := 10
  let d : ℤ := -2
  arithmetic_sequence a₁ d 10 = -8 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l755_75591


namespace NUMINAMATH_CALUDE_race_finish_count_l755_75539

/-- Calculates the number of men who finished a race given specific conditions --/
def men_finished_race (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped / 3
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 3
  let dehydrated_finished := dehydrated * 4 / 5
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 12 / 100
  let lost_finished := lost / 2
  let remaining_after_lost := remaining_after_dehydration - lost
  let faced_obstacle := remaining_after_lost * 3 / 8
  let obstacle_finished := faced_obstacle * 2 / 5
  tripped_finished + dehydrated_finished + lost_finished + obstacle_finished

/-- Theorem stating that given 80 men in the race, 41 men finished --/
theorem race_finish_count : men_finished_race 80 = 41 := by
  sorry

#eval men_finished_race 80

end NUMINAMATH_CALUDE_race_finish_count_l755_75539


namespace NUMINAMATH_CALUDE_rectangle_area_equals_50_l755_75561

/-- The area of a rectangle with height x and width 2x, whose perimeter is equal to the perimeter of an equilateral triangle with side length 10, is 50. -/
theorem rectangle_area_equals_50 : ∃ x : ℝ,
  let rectangle_height := x
  let rectangle_width := 2 * x
  let rectangle_perimeter := 2 * (rectangle_height + rectangle_width)
  let triangle_side_length := 10
  let triangle_perimeter := 3 * triangle_side_length
  let rectangle_area := rectangle_height * rectangle_width
  rectangle_perimeter = triangle_perimeter ∧ rectangle_area = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_50_l755_75561


namespace NUMINAMATH_CALUDE_john_remaining_money_l755_75537

def calculate_remaining_money (base_income : ℝ) (bonus_rate : ℝ) (transport_rate : ℝ)
  (rent : ℝ) (utilities : ℝ) (food : ℝ) (misc_rate : ℝ) (emergency_rate : ℝ)
  (retirement_rate : ℝ) (medical_expense : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_income := base_income * (1 + bonus_rate)
  let after_tax_income := total_income - (base_income * tax_rate)
  let fixed_expenses := rent + utilities + food
  let variable_expenses := (total_income * transport_rate) + (total_income * misc_rate)
  let savings_and_investments := (total_income * emergency_rate) + (total_income * retirement_rate)
  let total_expenses := fixed_expenses + variable_expenses + medical_expense + savings_and_investments
  after_tax_income - total_expenses

theorem john_remaining_money :
  calculate_remaining_money 2000 0.15 0.05 500 100 300 0.10 0.07 0.05 250 0.15 = 229 := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l755_75537


namespace NUMINAMATH_CALUDE_pencil_total_length_l755_75530

/-- The total length of a pencil with colored sections -/
def pencil_length (purple_length black_length blue_length : ℝ) : ℝ :=
  purple_length + black_length + blue_length

/-- Theorem: The total length of a pencil with specific colored sections is 4 cm -/
theorem pencil_total_length :
  pencil_length 1.5 0.5 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_total_length_l755_75530


namespace NUMINAMATH_CALUDE_max_spheres_in_specific_cylinder_l755_75575

/-- Represents a cylindrical container -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cylinder -/
def maxSpheresInCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

theorem max_spheres_in_specific_cylinder :
  let c := Cylinder.mk 82 225
  let s := Sphere.mk 38
  maxSpheresInCylinder c s = 21 := by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_specific_cylinder_l755_75575


namespace NUMINAMATH_CALUDE_complex_fractions_sum_l755_75599

theorem complex_fractions_sum (x y z : ℂ) 
  (h1 : x / (y + z) + y / (z + x) + z / (x + y) = 9)
  (h2 : x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y) = 64)
  (h3 : x^3 / (y + z) + y^3 / (z + x) + z^3 / (x + y) = 488) :
  x / (y * z) + y / (z * x) + z / (x * y) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fractions_sum_l755_75599


namespace NUMINAMATH_CALUDE_exhibition_planes_l755_75504

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- The number of planes in the exhibition -/
def num_planes : ℕ := total_wings / wings_per_plane

theorem exhibition_planes : num_planes = 25 := by
  sorry

end NUMINAMATH_CALUDE_exhibition_planes_l755_75504


namespace NUMINAMATH_CALUDE_paintedAreaPerimeter_is_260_l755_75522

/-- A framed artwork with given dimensions and border width. -/
structure FramedArtwork where
  outerWidth : ℝ
  outerHeight : ℝ
  borderWidth : ℝ

/-- Calculate the perimeter of the painted area in a framed artwork. -/
def paintedAreaPerimeter (art : FramedArtwork) : ℝ :=
  2 * ((art.outerWidth - 2 * art.borderWidth) + (art.outerHeight - 2 * art.borderWidth))

/-- Theorem: The perimeter of the painted area in the given framed artwork is 260 cm. -/
theorem paintedAreaPerimeter_is_260 :
  let art : FramedArtwork := {
    outerWidth := 100,
    outerHeight := 50,
    borderWidth := 5
  }
  paintedAreaPerimeter art = 260 := by
  sorry

end NUMINAMATH_CALUDE_paintedAreaPerimeter_is_260_l755_75522


namespace NUMINAMATH_CALUDE_F_3_f_4_equals_7_l755_75527

def f (a : ℝ) : ℝ := a - 2

def F (a b : ℝ) : ℝ := b^2 + a

theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_F_3_f_4_equals_7_l755_75527


namespace NUMINAMATH_CALUDE_total_gumballs_l755_75541

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine as described in the problem. -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧ m.green = 4 * m.blue ∧ m.red = 16

/-- Theorem stating that a valid gumball machine contains 56 gumballs in total. -/
theorem total_gumballs (m : GumballMachine) (h : validGumballMachine m) :
  m.red + m.blue + m.green = 56 := by
  sorry

#check total_gumballs

end NUMINAMATH_CALUDE_total_gumballs_l755_75541


namespace NUMINAMATH_CALUDE_tooth_arrangements_l755_75562

def word_length : Nat := 5
def repeated_letter_count : Nat := 2

theorem tooth_arrangements : 
  (word_length.factorial) / (repeated_letter_count.factorial * repeated_letter_count.factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tooth_arrangements_l755_75562


namespace NUMINAMATH_CALUDE_inequality_proof_l755_75549

theorem inequality_proof (t : ℝ) (h : 0 ≤ t ∧ t ≤ 6) : 
  Real.sqrt 6 ≤ Real.sqrt (-t + 6) + Real.sqrt t ∧ 
  Real.sqrt (-t + 6) + Real.sqrt t ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l755_75549


namespace NUMINAMATH_CALUDE_transport_probabilities_l755_75583

/-- Transportation method probabilities -/
structure TransportProb where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  sum_to_one : train + ship + car + airplane = 1
  all_nonneg : train ≥ 0 ∧ ship ≥ 0 ∧ car ≥ 0 ∧ airplane ≥ 0

/-- Given probabilities for each transportation method -/
def given_probs : TransportProb where
  train := 0.3
  ship := 0.1
  car := 0.2
  airplane := 0.4
  sum_to_one := by norm_num
  all_nonneg := by norm_num

/-- Theorem stating the probabilities of combined events -/
theorem transport_probabilities (p : TransportProb) :
  p.train + p.airplane = 0.7 ∧ 1 - p.ship = 0.9 := by sorry

end NUMINAMATH_CALUDE_transport_probabilities_l755_75583


namespace NUMINAMATH_CALUDE_sum_reciprocals_l755_75506

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (h1 : ω^4 = 1)
  (h2 : ω ≠ 1)
  (h3 : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l755_75506


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l755_75511

theorem polynomial_value_at_five : 
  let x : ℤ := 5
  x^5 - 3*x^3 - 5*x = 2725 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l755_75511


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l755_75566

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_digit_sum_property :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = sum_of_digits (1000^2) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l755_75566


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l755_75580

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l755_75580


namespace NUMINAMATH_CALUDE_g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l755_75589

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x - 1| + 2
def g (x : ℝ) : ℝ := -|x + 2| + 3

-- Theorem for the first part of the problem
theorem g_geq_neg_two_solution_set :
  {x : ℝ | g x ≥ -2} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for the second part of the problem
theorem f_minus_g_geq_m_plus_two_iff (m : ℝ) :
  (∀ x : ℝ, f x - g x ≥ m + 2) ↔ m ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l755_75589


namespace NUMINAMATH_CALUDE_three_digit_multiple_of_2_3_5_l755_75559

theorem three_digit_multiple_of_2_3_5 (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n →
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → 120 ≤ m) ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → m ≤ 990) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_multiple_of_2_3_5_l755_75559


namespace NUMINAMATH_CALUDE_track_width_l755_75525

theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 100 * π) →
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →
  (r₁ - r₂ = 8) := by
sorry

end NUMINAMATH_CALUDE_track_width_l755_75525


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l755_75568

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 19) / Real.sqrt (x^2 + 8) ≥ 2 * Real.sqrt 11 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 19) / Real.sqrt (x^2 + 8) = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l755_75568


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l755_75542

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l755_75542


namespace NUMINAMATH_CALUDE_f_on_negative_interval_l755_75594

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem f_on_negative_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : period_two f) 
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x := by
sorry

end NUMINAMATH_CALUDE_f_on_negative_interval_l755_75594


namespace NUMINAMATH_CALUDE_binomial_mode_maximizes_pmf_l755_75519

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of success -/
def p : ℚ := 3/4

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The mode of the binomial distribution -/
def binomialMode : ℕ := 4

/-- Theorem stating that the binomial mode maximizes the probability mass function -/
theorem binomial_mode_maximizes_pmf :
  ∀ k : ℕ, k ≤ n → binomialPMF binomialMode ≥ binomialPMF k :=
sorry

end NUMINAMATH_CALUDE_binomial_mode_maximizes_pmf_l755_75519


namespace NUMINAMATH_CALUDE_smaller_screen_diagonal_l755_75550

theorem smaller_screen_diagonal : 
  ∃ (x : ℝ), x > 0 ∧ x^2 + 34 = 18^2 ∧ x = Real.sqrt 290 := by sorry

end NUMINAMATH_CALUDE_smaller_screen_diagonal_l755_75550


namespace NUMINAMATH_CALUDE_f_increasing_on_neg_infinity_to_zero_l755_75523

-- Define the function f
def f (x : ℝ) : ℝ := 8 + 2*x - x^2

-- State the theorem
theorem f_increasing_on_neg_infinity_to_zero :
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_neg_infinity_to_zero_l755_75523


namespace NUMINAMATH_CALUDE_lcm_48_147_l755_75585

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_147_l755_75585


namespace NUMINAMATH_CALUDE_complex_equation_solution_l755_75521

theorem complex_equation_solution : ∃ (a : ℝ), 
  (∀ (i : ℂ), i * i = -1 → (a * i) / (2 - i) + 1 = 2 * i) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l755_75521


namespace NUMINAMATH_CALUDE_negation_equivalence_l755_75501

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 > 1) ↔
  (∀ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l755_75501


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l755_75590

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6)) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6) →
    m ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l755_75590


namespace NUMINAMATH_CALUDE_simplify_expression_l755_75547

theorem simplify_expression (a b c : ℝ) :
  -32 * a^4 * b^5 * c / (-2 * a * b)^3 * (-3/4 * a * c) = -3 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l755_75547


namespace NUMINAMATH_CALUDE_speed_with_400_people_l755_75592

/-- Represents the speed of a spaceship given the number of people on board. -/
def spaceshipSpeed (people : ℕ) : ℝ :=
  sorry

/-- The speed halves for every 100 additional people. -/
axiom speed_halves (n : ℕ) : spaceshipSpeed (n + 100) = (spaceshipSpeed n) / 2

/-- The speed of the spaceship with 200 people on board is 500 km/hr. -/
axiom initial_speed : spaceshipSpeed 200 = 500

/-- The speed of the spaceship with 400 people on board is 125 km/hr. -/
theorem speed_with_400_people : spaceshipSpeed 400 = 125 := by
  sorry

end NUMINAMATH_CALUDE_speed_with_400_people_l755_75592


namespace NUMINAMATH_CALUDE_quadratic_function_c_bounds_l755_75533

/-- Given a quadratic function f(x) = x² + bx + c, where b and c are real numbers,
    if 0 ≤ f(1) = f(2) ≤ 10, then 2 ≤ c ≤ 12 -/
theorem quadratic_function_c_bounds (b c : ℝ) :
  let f := fun x => x^2 + b*x + c
  (0 ≤ f 1) ∧ (f 1 = f 2) ∧ (f 2 ≤ 10) → 2 ≤ c ∧ c ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_c_bounds_l755_75533


namespace NUMINAMATH_CALUDE_product_bounds_l755_75540

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2+Real.sqrt 3)/8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l755_75540


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l755_75587

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 18) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 54 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l755_75587


namespace NUMINAMATH_CALUDE_multiples_of_15_between_25_and_225_l755_75520

theorem multiples_of_15_between_25_and_225 : 
  (Finset.range 226 
    |>.filter (fun n => n ≥ 25 ∧ n % 15 = 0)
    |>.card) = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_25_and_225_l755_75520


namespace NUMINAMATH_CALUDE_bear_food_consumption_l755_75588

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_in_three_weeks : ℝ := 15

/-- The number of weeks in the given condition -/
def given_weeks : ℝ := 3

/-- Theorem: For any number of weeks, the bear eats 5 times that many "Victors" worth of food -/
theorem bear_food_consumption (x : ℝ) : 
  (victors_in_three_weeks / given_weeks) * x = 5 * x := by
sorry

end NUMINAMATH_CALUDE_bear_food_consumption_l755_75588


namespace NUMINAMATH_CALUDE_flower_bunch_count_l755_75529

theorem flower_bunch_count (total_flowers : ℕ) (flowers_per_bunch : ℕ) (bunches : ℕ) : 
  total_flowers = 12 * 6 →
  flowers_per_bunch = 9 →
  bunches = total_flowers / flowers_per_bunch →
  bunches = 8 := by
sorry

end NUMINAMATH_CALUDE_flower_bunch_count_l755_75529


namespace NUMINAMATH_CALUDE_buddy_met_66_boys_l755_75593

/-- The number of girl students in the third grade -/
def num_girls : ℕ := 57

/-- The total number of third graders Buddy met -/
def total_students : ℕ := 123

/-- The number of boy students Buddy met -/
def num_boys : ℕ := total_students - num_girls

theorem buddy_met_66_boys : num_boys = 66 := by
  sorry

end NUMINAMATH_CALUDE_buddy_met_66_boys_l755_75593


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l755_75509

theorem binomial_coeff_not_coprime (n m k : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ∃ d : ℕ, d > 1 ∧ d ∣ Nat.choose n k ∧ d ∣ Nat.choose n m :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l755_75509


namespace NUMINAMATH_CALUDE_square_divides_power_plus_one_l755_75557

theorem square_divides_power_plus_one (n : ℕ) : n^2 ∣ 2^n + 1 ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_divides_power_plus_one_l755_75557


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l755_75544

theorem sqrt_sum_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt a + Real.sqrt b ≥ Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l755_75544


namespace NUMINAMATH_CALUDE_prime_saturated_bound_l755_75534

def isPrimeSaturated (n : ℕ) (bound : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < bound

def isGreatestTwoDigitPrimeSaturated (n : ℕ) : Prop :=
  n ≤ 99 ∧ isPrimeSaturated n 96 ∧ ∀ m, m ≤ 99 → isPrimeSaturated m 96 → m ≤ n

theorem prime_saturated_bound (n : ℕ) :
  isGreatestTwoDigitPrimeSaturated 96 →
  isPrimeSaturated n (Finset.prod (Nat.factors n).toFinset id + 1) →
  Finset.prod (Nat.factors n).toFinset id < 96 :=
by sorry

end NUMINAMATH_CALUDE_prime_saturated_bound_l755_75534


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l755_75555

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 108 := by
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l755_75555


namespace NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l755_75500

/-- An arithmetic sequence with non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n, a (n + 1) - a n = d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, b (n + 1) / b n = q

theorem geometric_sequence_from_arithmetic (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  b 2 = 5 →
  a 5 = b 1 →
  a 8 = b 2 →
  a 13 = b 3 →
  ∀ n, b n = 3 * (5/3)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l755_75500


namespace NUMINAMATH_CALUDE_triangle_inequality_l755_75513

theorem triangle_inequality (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : A + B + C = π) : 
  (Real.sin (2*A) + Real.sin (2*B))^2 / (Real.sin A * Real.sin B) + 
  (Real.sin (2*B) + Real.sin (2*C))^2 / (Real.sin B * Real.sin C) + 
  (Real.sin (2*C) + Real.sin (2*A))^2 / (Real.sin C * Real.sin A) ≤ 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l755_75513


namespace NUMINAMATH_CALUDE_abc_value_l755_75517

theorem abc_value (a b c : ℂ) 
  (eq1 : 2 * a * b + 3 * b = -21)
  (eq2 : 2 * b * c + 3 * c = -21)
  (eq3 : 2 * c * a + 3 * a = -21) :
  a * b * c = 105.75 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l755_75517


namespace NUMINAMATH_CALUDE_divisors_of_60_and_84_l755_75558

theorem divisors_of_60_and_84 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ d : ℕ, d > 0 ∧ (60 % d = 0 ∧ 84 % d = 0) ↔ d ∈ Finset.range n) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_60_and_84_l755_75558


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l755_75564

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l755_75564


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l755_75546

theorem nested_expression_evaluation : (3*(3*(4*(3*(4*(2+1)+1)+2)+1)+2)+1) = 1492 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l755_75546


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l755_75582

theorem point_in_second_quadrant (m : ℝ) :
  (m - 1 < 0 ∧ 3 > 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l755_75582


namespace NUMINAMATH_CALUDE_sqrt_expressions_l755_75576

theorem sqrt_expressions (x y : ℝ) 
  (hx : x = Real.sqrt 3 + Real.sqrt 2) 
  (hy : y = Real.sqrt 3 - Real.sqrt 2) : 
  x^2 + 2*x*y + y^2 = 12 ∧ 1/y - 1/x = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l755_75576


namespace NUMINAMATH_CALUDE_policeman_hats_l755_75581

theorem policeman_hats (simpson_hats : ℕ) (obrien_hats_after : ℕ) : 
  simpson_hats = 15 →
  obrien_hats_after = 34 →
  ∃ (obrien_hats_before : ℕ), 
    obrien_hats_before > 2 * simpson_hats ∧
    obrien_hats_before = obrien_hats_after + 1 ∧
    obrien_hats_before - 2 * simpson_hats = 5 :=
by sorry

end NUMINAMATH_CALUDE_policeman_hats_l755_75581


namespace NUMINAMATH_CALUDE_hat_markup_price_l755_75548

theorem hat_markup_price (P : ℝ) 
  (h1 : 2 * P - (P + 0.7 * P) = 6) : 
  P + 0.7 * P = 34 := by
  sorry

end NUMINAMATH_CALUDE_hat_markup_price_l755_75548


namespace NUMINAMATH_CALUDE_diag_diff_octagon_heptagon_l755_75516

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of diagonals in a heptagon -/
def A : ℕ := num_diagonals 7

/-- Number of diagonals in an octagon -/
def B : ℕ := num_diagonals 8

/-- The difference between the number of diagonals in an octagon and a heptagon is 6 -/
theorem diag_diff_octagon_heptagon : B - A = 6 := by sorry

end NUMINAMATH_CALUDE_diag_diff_octagon_heptagon_l755_75516


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l755_75573

def polynomial (x : ℝ) : ℝ := (x^2 - 3*x + 2) * x * (x - 4)

theorem roots_of_polynomial :
  {x : ℝ | polynomial x = 0} = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l755_75573


namespace NUMINAMATH_CALUDE_total_students_correct_l755_75515

/-- The total number of students at the college -/
def total_students : ℝ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_percentage : ℝ := 32.5

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 594

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_percentage / 100) * total_students = non_biology_students :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l755_75515


namespace NUMINAMATH_CALUDE_gabriel_jaxon_toy_ratio_l755_75556

theorem gabriel_jaxon_toy_ratio :
  ∀ (g j x : ℕ),
  j = g + 8 →
  x = 15 →
  g + j + x = 83 →
  g = 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_gabriel_jaxon_toy_ratio_l755_75556


namespace NUMINAMATH_CALUDE_remaining_juice_bottles_l755_75577

/-- Given the initial number of juice bottles in the refrigerator and pantry,
    the number of bottles bought, and the number of bottles consumed,
    calculate the remaining number of bottles. -/
theorem remaining_juice_bottles
  (refrigerator_bottles : ℕ)
  (pantry_bottles : ℕ)
  (bought_bottles : ℕ)
  (consumed_bottles : ℕ)
  (h1 : refrigerator_bottles = 4)
  (h2 : pantry_bottles = 4)
  (h3 : bought_bottles = 5)
  (h4 : consumed_bottles = 3) :
  refrigerator_bottles + pantry_bottles + bought_bottles - consumed_bottles = 10 := by
  sorry

#check remaining_juice_bottles

end NUMINAMATH_CALUDE_remaining_juice_bottles_l755_75577


namespace NUMINAMATH_CALUDE_gear_rotation_l755_75553

theorem gear_rotation (teeth_A teeth_B turns_A : ℕ) (h1 : teeth_A = 6) (h2 : teeth_B = 8) (h3 : turns_A = 12) :
  teeth_A * turns_A = teeth_B * (teeth_A * turns_A / teeth_B) :=
sorry

end NUMINAMATH_CALUDE_gear_rotation_l755_75553


namespace NUMINAMATH_CALUDE_logarithm_and_exponential_equalities_l755_75552

theorem logarithm_and_exponential_equalities :
  (Real.log 9 / Real.log 6 + 2 * Real.log 2 / Real.log 6 = 2) ∧
  (Real.exp 0 + Real.sqrt ((1 - Real.sqrt 2)^2) - 8^(1/6) = 1 + Real.sqrt 5 - Real.sqrt 2 - 2^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_and_exponential_equalities_l755_75552


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_implies_radius_three_l755_75578

theorem sphere_volume_equal_surface_area_implies_radius_three 
  (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_implies_radius_three_l755_75578


namespace NUMINAMATH_CALUDE_abc_sum_product_l755_75572

theorem abc_sum_product (x : ℝ) : ∃ a b c : ℝ, a + b + c = 1 ∧ a * b + a * c + b * c = x := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_l755_75572


namespace NUMINAMATH_CALUDE_prob_b_is_three_fourths_l755_75531

/-- The probability that either A or B solves a problem, given their individual probabilities -/
def prob_either_solves (prob_a prob_b : ℝ) : ℝ :=
  prob_a + prob_b - prob_a * prob_b

/-- Theorem stating that if A's probability is 2/3 and the probability of either A or B solving
    is 0.9166666666666666, then B's probability is 3/4 -/
theorem prob_b_is_three_fourths (prob_a prob_b : ℝ) 
    (h1 : prob_a = 2/3)
    (h2 : prob_either_solves prob_a prob_b = 0.9166666666666666) :
    prob_b = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_prob_b_is_three_fourths_l755_75531


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l755_75536

open Complex

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (I + 1) = 2 / (I - 1)) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l755_75536


namespace NUMINAMATH_CALUDE_f_is_quadratic_l755_75551

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 4 = 4 -/
def f (x : ℝ) : ℝ := x^2 - 8

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l755_75551


namespace NUMINAMATH_CALUDE_geralds_initial_notebooks_l755_75543

theorem geralds_initial_notebooks (jack_initial gerald_initial jack_remaining paula_given mike_given : ℕ) : 
  jack_initial = gerald_initial + 13 →
  jack_initial = jack_remaining + paula_given + mike_given →
  jack_remaining = 10 →
  paula_given = 5 →
  mike_given = 6 →
  gerald_initial = 8 := by
sorry

end NUMINAMATH_CALUDE_geralds_initial_notebooks_l755_75543


namespace NUMINAMATH_CALUDE_power_function_value_l755_75503

theorem power_function_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x ^ a) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l755_75503


namespace NUMINAMATH_CALUDE_find_a_l755_75545

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 3 * a * x^2 + 6 * x) ∧ (deriv (f a)) (-1) = 3 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l755_75545


namespace NUMINAMATH_CALUDE_jessica_initial_money_l755_75505

/-- The amount of money Jessica spent on a cat toy -/
def spent : ℚ := 10.22

/-- The amount of money Jessica has left -/
def left : ℚ := 1.51

/-- Jessica's initial amount of money -/
def initial : ℚ := spent + left

/-- Theorem stating that Jessica's initial amount of money was $11.73 -/
theorem jessica_initial_money : initial = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_jessica_initial_money_l755_75505


namespace NUMINAMATH_CALUDE_trip_distance_l755_75528

theorem trip_distance (speed1 speed2 time_saved : ℝ) (h1 : speed1 = 50) (h2 : speed2 = 60) (h3 : time_saved = 4) :
  let distance := speed1 * speed2 * time_saved / (speed2 - speed1)
  distance = 1200 := by sorry

end NUMINAMATH_CALUDE_trip_distance_l755_75528


namespace NUMINAMATH_CALUDE_amys_garden_space_l755_75554

/-- Calculates the total square footage of growing space for Amy's garden beds -/
theorem amys_garden_space (small_bed_length small_bed_width : ℝ)
                           (large_bed_length large_bed_width : ℝ)
                           (num_small_beds num_large_beds : ℕ) :
  small_bed_length = 3 →
  small_bed_width = 3 →
  large_bed_length = 4 →
  large_bed_width = 3 →
  num_small_beds = 2 →
  num_large_beds = 2 →
  (num_small_beds : ℝ) * (small_bed_length * small_bed_width) +
  (num_large_beds : ℝ) * (large_bed_length * large_bed_width) = 42 := by
  sorry

#check amys_garden_space

end NUMINAMATH_CALUDE_amys_garden_space_l755_75554


namespace NUMINAMATH_CALUDE_zoe_strawberry_count_l755_75596

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := sorry

/-- The number of ounces of yogurt Zoe ate -/
def yogurt_ounces : ℕ := 6

/-- Calories per strawberry -/
def calories_per_strawberry : ℕ := 4

/-- Calories per ounce of yogurt -/
def calories_per_yogurt_ounce : ℕ := 17

/-- Total calories consumed -/
def total_calories : ℕ := 150

theorem zoe_strawberry_count :
  num_strawberries * calories_per_strawberry +
  yogurt_ounces * calories_per_yogurt_ounce = total_calories ∧
  num_strawberries = 12 := by sorry

end NUMINAMATH_CALUDE_zoe_strawberry_count_l755_75596


namespace NUMINAMATH_CALUDE_w_squared_value_l755_75570

theorem w_squared_value (w : ℝ) (h : (2*w + 10)^2 = (5*w + 15)*(w + 6)) : 
  w^2 = (90 + 10*Real.sqrt 65) / 4 := by
sorry

end NUMINAMATH_CALUDE_w_squared_value_l755_75570


namespace NUMINAMATH_CALUDE_sharpshooter_target_orders_l755_75538

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multisetPermutations (total : ℕ) (counts : List ℕ) : ℕ :=
  factorial total / (counts.map factorial).prod

theorem sharpshooter_target_orders : 
  let total_targets : ℕ := 8
  let column_targets : List ℕ := [2, 3, 2, 1]
  multisetPermutations total_targets column_targets = 1680 := by
  sorry

end NUMINAMATH_CALUDE_sharpshooter_target_orders_l755_75538


namespace NUMINAMATH_CALUDE_bank_deposit_l755_75512

theorem bank_deposit (P : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) : 
  interest_rate = 0.1 →
  years = 2 →
  final_amount = 121 →
  P * (1 + interest_rate) ^ years = final_amount →
  P = 100 := by
sorry

end NUMINAMATH_CALUDE_bank_deposit_l755_75512


namespace NUMINAMATH_CALUDE_remainder_of_587421_div_6_l755_75571

theorem remainder_of_587421_div_6 : 587421 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_587421_div_6_l755_75571


namespace NUMINAMATH_CALUDE_sum_of_composite_function_at_specific_points_l755_75598

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := |x + 1| - 3

def x_values : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_composite_function_at_specific_points :
  (x_values.map (λ x => q (p x))).sum = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_at_specific_points_l755_75598


namespace NUMINAMATH_CALUDE_gina_charity_fraction_l755_75574

def initial_amount : ℚ := 400
def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def kept_amount : ℚ := 170
def charity_fraction : ℚ := 1/5

theorem gina_charity_fraction :
  charity_fraction = (initial_amount - mom_fraction * initial_amount - clothes_fraction * initial_amount - kept_amount) / initial_amount := by
  sorry

end NUMINAMATH_CALUDE_gina_charity_fraction_l755_75574


namespace NUMINAMATH_CALUDE_shirt_sale_profit_l755_75535

theorem shirt_sale_profit (total_shirts : ℕ) (total_cost : ℕ) 
  (black_wholesale : ℕ) (white_wholesale : ℕ) 
  (black_retail : ℕ) (white_retail : ℕ) :
  total_shirts = 200 →
  total_cost = 3500 →
  black_wholesale = 25 →
  white_wholesale = 15 →
  black_retail = 50 →
  white_retail = 35 →
  ∃ (black_count white_count : ℕ),
    black_count = 50 ∧
    white_count = 150 ∧
    black_count + white_count = total_shirts ∧
    black_count * black_wholesale + white_count * white_wholesale = total_cost ∧
    (black_count * (black_retail - black_wholesale) + 
     white_count * (white_retail - white_wholesale)) = 4250 :=
by sorry

end NUMINAMATH_CALUDE_shirt_sale_profit_l755_75535


namespace NUMINAMATH_CALUDE_cafeteria_apples_l755_75586

/-- The number of apples in the school cafeteria after using some for lunch and buying more. -/
def final_apples (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

/-- Theorem stating that given the specific numbers in the problem, the final number of apples is 9. -/
theorem cafeteria_apples : final_apples 23 20 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l755_75586


namespace NUMINAMATH_CALUDE_correct_volunteer_assignment_l755_75569

/-- The number of ways to select and assign volunteers to tour groups --/
def assignVolunteers (totalVolunteers femaleVolunteers tourGroups : ℕ) : ℕ :=
  let maleVolunteers := totalVolunteers - femaleVolunteers
  let totalCombinations := Nat.choose totalVolunteers tourGroups
  let allFemaleCombinations := Nat.choose femaleVolunteers tourGroups
  let allMaleCombinations := Nat.choose maleVolunteers tourGroups
  let validCombinations := totalCombinations - allFemaleCombinations - allMaleCombinations
  validCombinations * Nat.factorial tourGroups

/-- Theorem stating the correct number of ways to assign volunteers --/
theorem correct_volunteer_assignment :
  assignVolunteers 10 4 3 = 576 :=
by sorry

end NUMINAMATH_CALUDE_correct_volunteer_assignment_l755_75569


namespace NUMINAMATH_CALUDE_prop_2_prop_3_l755_75584

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are distinct
variable (h_distinct_lines : m ≠ n)

-- State that α and β are different
variable (h_different_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (parallel_planes α β ∧ subset m α) → parallel_lines m β :=
sorry

-- Proposition ③
theorem prop_3 : 
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α) → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_l755_75584
