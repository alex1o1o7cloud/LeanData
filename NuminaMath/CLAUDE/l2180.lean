import Mathlib

namespace product_inequality_l2180_218098

theorem product_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_condition : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end product_inequality_l2180_218098


namespace custom_calculator_results_l2180_218051

/-- A custom operation that satisfies specific properties -/
noncomputable def customOp (a b : ℕ) : ℕ :=
  sorry

/-- Addition operation -/
def add : ℕ → ℕ → ℕ := (·+·)

axiom custom_op_self (a : ℕ) : customOp a a = a

axiom custom_op_zero (a : ℕ) : customOp a 0 = 2 * a

axiom custom_op_distributive (a b c d : ℕ) :
  add (customOp a b) (customOp c d) = add (customOp a c) (customOp b d)

theorem custom_calculator_results :
  (customOp (add 2 3) (add 0 3) = 7) ∧
  (customOp 1024 48 = 2000) := by
  sorry

end custom_calculator_results_l2180_218051


namespace line_parallel_perpendicular_implies_planes_perpendicular_l2180_218050

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the parallel and perpendicular relations
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : L) (α β : P) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l2180_218050


namespace volume_of_intersected_prism_l2180_218018

/-- The volume of a solid formed by the intersection of a plane with a prism -/
theorem volume_of_intersected_prism (a : ℝ) (h : ℝ) :
  let prism_base_area : ℝ := (a^2 * Real.sqrt 3) / 2
  let prism_volume : ℝ := prism_base_area * h
  let intersection_volume : ℝ := (77 * Real.sqrt 3) / 54
  (h = 2) →
  (intersection_volume < prism_volume) →
  (intersection_volume > 0) →
  intersection_volume = (77 * Real.sqrt 3) / 54 :=
by sorry

end volume_of_intersected_prism_l2180_218018


namespace inverse_square_theorem_l2180_218048

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop := ∃ k : ℝ, x = k / (y * y)

-- Define the theorem
theorem inverse_square_theorem :
  ∀ x y : ℝ,
  inverse_square_relation x y →
  (9 : ℝ) * (9 : ℝ) * (0.1111111111111111 : ℝ) = (3 : ℝ) * (3 : ℝ) * (1 : ℝ) →
  (x = (1 : ℝ) → y = (3 : ℝ)) :=
by sorry

end inverse_square_theorem_l2180_218048


namespace whatsapp_messages_l2180_218061

/-- The number of messages sent in a Whatsapp group over four days -/
def total_messages (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given the conditions of the Whatsapp group messages, 
    the total number of messages over four days is 2000 -/
theorem whatsapp_messages : 
  ∀ (monday tuesday wednesday thursday : ℕ),
    monday = 300 →
    tuesday = 200 →
    wednesday = tuesday + 300 →
    thursday = 2 * wednesday →
    total_messages monday tuesday wednesday thursday = 2000 :=
by
  sorry


end whatsapp_messages_l2180_218061


namespace power_inequality_l2180_218077

theorem power_inequality (p q a : ℝ) (h1 : p > q) (h2 : q > 1) (h3 : 0 < a) (h4 : a < 1) :
  p ^ a > q ^ a := by
  sorry

end power_inequality_l2180_218077


namespace hammond_marble_weight_l2180_218031

/-- The weight of Hammond's marble statues and discarded marble -/
structure MarbleStatues where
  first_statue : ℕ
  second_statue : ℕ
  remaining_statues : ℕ
  discarded_marble : ℕ

/-- The initial weight of the marble block -/
def initial_weight (m : MarbleStatues) : ℕ :=
  m.first_statue + m.second_statue + 2 * m.remaining_statues + m.discarded_marble

/-- Theorem stating the initial weight of Hammond's marble block -/
theorem hammond_marble_weight :
  ∃ (m : MarbleStatues),
    m.first_statue = 10 ∧
    m.second_statue = 18 ∧
    m.remaining_statues = 15 ∧
    m.discarded_marble = 22 ∧
    initial_weight m = 80 := by
  sorry

end hammond_marble_weight_l2180_218031


namespace height_of_cone_l2180_218078

/-- Theorem: Height of a cone with specific volume and vertex angle -/
theorem height_of_cone (V : ℝ) (angle : ℝ) (h : ℝ) :
  V = 16384 * Real.pi ∧ angle = 90 →
  h = (49152 : ℝ) ^ (1/3) :=
by sorry

end height_of_cone_l2180_218078


namespace picasso_paintings_probability_l2180_218073

/-- The probability of placing 4 Picasso paintings consecutively among 12 art pieces -/
theorem picasso_paintings_probability (total_pieces : ℕ) (picasso_paintings : ℕ) :
  total_pieces = 12 →
  picasso_paintings = 4 →
  (picasso_paintings.factorial * (total_pieces - picasso_paintings + 1).factorial) / total_pieces.factorial = 1 / 55 :=
by sorry

end picasso_paintings_probability_l2180_218073


namespace units_digit_of_2749_pow_987_l2180_218044

theorem units_digit_of_2749_pow_987 :
  (2749^987) % 10 = 9 := by
  sorry

end units_digit_of_2749_pow_987_l2180_218044


namespace triangle_inequality_l2180_218010

theorem triangle_inequality (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b →
  ¬(a = 3 ∧ b = 5 ∧ c = 2) := by
  sorry

end triangle_inequality_l2180_218010


namespace division_simplification_l2180_218064

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^4 * y^2 / (-2 * x * y) = -2 * x^3 * y :=
by sorry

end division_simplification_l2180_218064


namespace system_of_equations_solution_l2180_218092

theorem system_of_equations_solution : 
  let x : ℚ := -29/2
  let y : ℚ := -71/2
  (7 * x - 3 * y = 5) ∧ (y - 3 * x = 8) := by
sorry

end system_of_equations_solution_l2180_218092


namespace line_properties_l2180_218023

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ

/-- The theorem stating the properties of the given parameterized line -/
theorem line_properties (L : ParameterizedLine) : 
  L.point 1 = (2, 5) ∧ L.point 4 = (5, -7) → L.point 0 = (1, 9) := by
  sorry


end line_properties_l2180_218023


namespace average_speed_calculation_l2180_218074

-- Define the sections of the trip
def section1_distance : ℝ := 600
def section1_speed : ℝ := 30
def section2_distance : ℝ := 300
def section2_speed : ℝ := 15
def section3_distance : ℝ := 500
def section3_speed : ℝ := 25
def section4_distance : ℝ := 400
def section4_speed : ℝ := 40

-- Define the total distance
def total_distance : ℝ := section1_distance + section2_distance + section3_distance + section4_distance

-- Theorem statement
theorem average_speed_calculation :
  let time1 := section1_distance / section1_speed
  let time2 := section2_distance / section2_speed
  let time3 := section3_distance / section3_speed
  let time4 := section4_distance / section4_speed
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  abs (average_speed - 25.71) < 0.01 := by
  sorry

end average_speed_calculation_l2180_218074


namespace symmetry_axis_shifted_even_function_l2180_218014

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to have an axis of symmetry
def has_axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_axis_shifted_even_function :
  is_even (λ x => f (x + 2)) → has_axis_of_symmetry f 2 :=
by sorry

end symmetry_axis_shifted_even_function_l2180_218014


namespace game_question_count_l2180_218036

theorem game_question_count (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end game_question_count_l2180_218036


namespace value_of_P_closed_under_multiplication_l2180_218008

/-- The polynomial P(x, y) = 2x^2 - 6xy + 5y^2 -/
def P (x y : ℤ) : ℤ := 2*x^2 - 6*x*y + 5*y^2

/-- A number is a value of P if it can be expressed as P(b, c) for some integers b and c -/
def is_value_of_P (a : ℤ) : Prop := ∃ b c : ℤ, P b c = a

/-- If r and s are values of P, then rs is also a value of P -/
theorem value_of_P_closed_under_multiplication (r s : ℤ) 
  (hr : is_value_of_P r) (hs : is_value_of_P s) : 
  is_value_of_P (r * s) := by
  sorry

end value_of_P_closed_under_multiplication_l2180_218008


namespace ali_seashells_to_friends_l2180_218049

/-- The number of seashells Ali gave to his friends -/
def seashells_to_friends (initial : ℕ) (to_brothers : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_brothers - 2 * remaining

theorem ali_seashells_to_friends :
  seashells_to_friends 180 30 55 = 40 := by
  sorry

end ali_seashells_to_friends_l2180_218049


namespace total_handshakes_l2180_218060

/-- The number of teams in the tournament -/
def num_teams : ℕ := 3

/-- The number of players in each team -/
def players_per_team : ℕ := 4

/-- The number of referees -/
def num_referees : ℕ := 3

/-- The number of coaches -/
def num_coaches : ℕ := 1

/-- The total number of players -/
def total_players : ℕ := num_teams * players_per_team

/-- The number of officials (referees + coaches) -/
def num_officials : ℕ := num_referees + num_coaches

/-- Theorem stating the total number of handshakes in the tournament -/
theorem total_handshakes : 
  (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
  (total_players * num_officials) = 144 := by
  sorry

#eval (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
      (total_players * num_officials)

end total_handshakes_l2180_218060


namespace find_n_value_l2180_218096

theorem find_n_value (n : ℕ) : (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10 : ℝ)^35) → n = 35 := by
  sorry

end find_n_value_l2180_218096


namespace volunteer_arrangements_l2180_218062

def num_applicants : ℕ := 5
def num_selected : ℕ := 3
def num_events : ℕ := 3

def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

theorem volunteer_arrangements : 
  permutations num_applicants num_selected - 
  permutations (num_applicants - 1) (num_selected - 1) = 48 := by
  sorry

end volunteer_arrangements_l2180_218062


namespace root_value_theorem_l2180_218028

theorem root_value_theorem (m : ℝ) : m^2 - 6*m - 5 = 0 → 11 + 6*m - m^2 = 6 := by
  sorry

end root_value_theorem_l2180_218028


namespace megan_files_added_l2180_218047

theorem megan_files_added 
  (initial_files : ℝ) 
  (files_per_folder : ℝ) 
  (num_folders : ℝ) 
  (h1 : initial_files = 93.0) 
  (h2 : files_per_folder = 8.0) 
  (h3 : num_folders = 14.25) : 
  num_folders * files_per_folder - initial_files = 21.0 := by
sorry

end megan_files_added_l2180_218047


namespace wanda_crayon_count_l2180_218076

/-- The number of crayons Wanda, Dina, and Jacob have. -/
structure CrayonCount where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ

/-- The given conditions for the crayon problem. -/
def crayon_problem (c : CrayonCount) : Prop :=
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.wanda + c.dina + c.jacob = 116

/-- Theorem stating that Wanda has 62 crayons given the conditions. -/
theorem wanda_crayon_count (c : CrayonCount) (h : crayon_problem c) : c.wanda = 62 := by
  sorry

end wanda_crayon_count_l2180_218076


namespace problem_statement_l2180_218065

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (1 - 2*m) + y^2 / (m + 2) = 1 ∧ (1 - 2*m) * (m + 2) < 0

theorem problem_statement :
  (∀ m : ℝ, p m ↔ (m ≤ -1 ∨ m ≥ 2)) ∧
  (∀ m : ℝ, q m ↔ (m < -2 ∨ m > 1/2)) ∧
  (∀ m : ℝ, ¬(p m ∨ q m) ↔ (-1 < m ∧ m ≤ 1/2)) := by sorry

end problem_statement_l2180_218065


namespace taras_birthday_money_l2180_218066

theorem taras_birthday_money (P : ℝ) : P * 1.1 = 99 → P = 90 := by
  sorry

end taras_birthday_money_l2180_218066


namespace arrangement_count_is_correct_l2180_218095

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three consecutive balls are painted red -/
def arrangementCount : ℕ := 24

/-- The total number of balls -/
def totalBalls : ℕ := 8

/-- The number of red balls -/
def redBalls : ℕ := 5

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutiveRedBalls : ℕ := 3

theorem arrangement_count_is_correct :
  arrangementCount = 24 ∧
  totalBalls = 8 ∧
  redBalls = 5 ∧
  whiteBalls = 3 ∧
  consecutiveRedBalls = 3 ∧
  redBalls + whiteBalls = totalBalls ∧
  arrangementCount = (whiteBalls + 1) * (redBalls - consecutiveRedBalls + 1) :=
by sorry

end arrangement_count_is_correct_l2180_218095


namespace hex_numeric_count_2023_l2180_218094

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits (0-9) --/
def hex_only_numeric (l : List (Fin 16)) : Bool :=
  sorry

/-- Counts numbers up to n whose hexadecimal representation contains only numeric digits --/
def count_hex_numeric (n : ℕ+) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sum_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem statement --/
theorem hex_numeric_count_2023 :
  sum_digits (count_hex_numeric 2023) = 25 :=
sorry

end hex_numeric_count_2023_l2180_218094


namespace ball_distribution_count_l2180_218099

theorem ball_distribution_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let ways_per_ball : ℕ := n_boxes
  n_boxes ^ n_balls = 81 :=
by sorry

end ball_distribution_count_l2180_218099


namespace simplify_fraction_l2180_218009

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l2180_218009


namespace max_height_formula_l2180_218039

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum possible height of the table formed by right angle folds -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The specific triangle in the problem -/
def problem_triangle : Triangle := { a := 25, b := 28, c := 31 }

theorem max_height_formula : 
  max_table_height problem_triangle = 42 * Real.sqrt 2582 / 28 := by sorry

end max_height_formula_l2180_218039


namespace sock_drawer_probability_l2180_218053

/-- The total number of socks in the drawer -/
def total_socks : ℕ := 2016

/-- The number of copper socks -/
def copper_socks : ℕ := 2000

/-- The number of colors other than copper -/
def other_colors : ℕ := 8

/-- The number of socks for each color other than copper -/
def socks_per_color : ℕ := 2

/-- The probability of drawing two socks of the same color or one red and one green sock -/
def probability : ℚ := 1999012 / 2031120

theorem sock_drawer_probability :
  (copper_socks.choose 2 + other_colors * socks_per_color.choose 2 + socks_per_color ^ 2) /
  total_socks.choose 2 = probability := by sorry

end sock_drawer_probability_l2180_218053


namespace square_of_nines_l2180_218088

theorem square_of_nines (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 := by
  sorry

end square_of_nines_l2180_218088


namespace clerk_forms_per_hour_l2180_218005

theorem clerk_forms_per_hour 
  (total_forms : ℕ) 
  (work_hours : ℕ) 
  (num_clerks : ℕ) 
  (h1 : total_forms = 2400) 
  (h2 : work_hours = 8) 
  (h3 : num_clerks = 12) : 
  (total_forms / work_hours) / num_clerks = 25 := by
sorry

end clerk_forms_per_hour_l2180_218005


namespace sum_of_a_n_equals_1158_l2180_218082

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 10 = 0 then 12
  else if n % 10 = 0 ∧ n % 9 = 0 then 15
  else if n % 9 = 0 ∧ n % 15 = 0 then 10
  else 0

theorem sum_of_a_n_equals_1158 :
  (Finset.range 1499).sum a = 1158 := by
  sorry

end sum_of_a_n_equals_1158_l2180_218082


namespace rectangle_measurement_error_l2180_218089

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_W : W > 0) :
  (1.16 * L) * (W * (1 - x / 100)) = 1.102 * (L * W) → x = 5 := by
  sorry

end rectangle_measurement_error_l2180_218089


namespace sin_cos_equation_solutions_l2180_218013

theorem sin_cos_equation_solutions (x : ℝ) :
  (0 ≤ x ∧ x < 2 * Real.pi) ∧ (Real.sin x - Real.cos x = Real.sqrt 3 / 2) ↔
  (x = Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4 ∨
   x = Real.pi - Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4) :=
by sorry

end sin_cos_equation_solutions_l2180_218013


namespace segment_intersection_theorem_l2180_218007

/-- Represents a line in the real plane -/
structure Line where
  -- Add necessary fields

/-- Represents a line segment in the real plane -/
structure Segment where
  -- Add necessary fields

/-- Predicate to check if a line intersects a segment -/
def intersects (l : Line) (s : Segment) : Prop :=
  sorry

/-- Predicate to check if segments are concurrent -/
def concurrent (segments : List Segment) : Prop :=
  sorry

theorem segment_intersection_theorem
  (n : ℕ)
  (segments : List Segment)
  (h_concurrent : concurrent segments)
  (h_count : segments.length = n)
  (h_triple_intersection : ∀ (s1 s2 s3 : Segment),
    s1 ∈ segments → s2 ∈ segments → s3 ∈ segments →
    ∃ (l : Line), intersects l s1 ∧ intersects l s2 ∧ intersects l s3) :
  ∃ (l : Line), ∀ (s : Segment), s ∈ segments → intersects l s :=
sorry

end segment_intersection_theorem_l2180_218007


namespace sqrt_of_neg_nine_l2180_218000

-- Define the square root of a complex number
def complex_sqrt (z : ℂ) : Set ℂ :=
  {w : ℂ | w^2 = z}

-- Theorem statement
theorem sqrt_of_neg_nine :
  complex_sqrt (-9 : ℂ) = {3*I, -3*I} :=
sorry

end sqrt_of_neg_nine_l2180_218000


namespace max_sum_of_squares_l2180_218026

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 52 →
  a * d + b * c = 83 →
  c * d = 42 →
  a^2 + b^2 + c^2 + d^2 ≤ 38 :=
by sorry

end max_sum_of_squares_l2180_218026


namespace correct_result_l2180_218075

theorem correct_result (x : ℝ) (h : x / 3 = 45) : 3 * x = 405 := by
  sorry

end correct_result_l2180_218075


namespace special_sequence_sum_5_l2180_218033

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  roots_property : a 2 * a 4 = 3 ∧ a 2 + a 4 = 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : SpecialArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem: S_5 = 5/2 for the special arithmetic sequence -/
theorem special_sequence_sum_5 (seq : SpecialArithmeticSequence) : 
  sum_n seq 5 = 5 / 2 := by
  sorry

end special_sequence_sum_5_l2180_218033


namespace quadratic_inequalities_solution_sets_l2180_218067

theorem quadratic_inequalities_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  Set.Ioo (-1/2) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
sorry

end quadratic_inequalities_solution_sets_l2180_218067


namespace terminal_sides_theorem_l2180_218056

/-- Given an angle θ in degrees, returns true if the terminal side of 7θ coincides with the terminal side of θ -/
def terminal_sides_coincide (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + k * 360

/-- The set of angles whose terminal sides coincide with their 7θ counterparts -/
def coinciding_angles : Set ℝ := {0, 60, 120, 180, 240, 300}

theorem terminal_sides_theorem (θ : ℝ) :
  0 ≤ θ ∧ θ < 360 ∧ terminal_sides_coincide θ → θ ∈ coinciding_angles := by
  sorry

end terminal_sides_theorem_l2180_218056


namespace burger_orders_l2180_218097

theorem burger_orders (total : ℕ) (burger_ratio : ℕ) : 
  total = 45 → burger_ratio = 2 → 
  ∃ (hotdog : ℕ), 
    hotdog + burger_ratio * hotdog = total ∧
    burger_ratio * hotdog = 30 := by
  sorry

end burger_orders_l2180_218097


namespace scientific_notation_equality_l2180_218059

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000012 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = -7 := by
  sorry

end scientific_notation_equality_l2180_218059


namespace min_reciprocal_sum_l2180_218085

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 1) :
  (1/a + 1/b) ≥ 455/36 :=
by sorry

end min_reciprocal_sum_l2180_218085


namespace max_rectangles_l2180_218001

/-- Represents a cell in the figure -/
inductive Cell
| White
| Black

/-- Represents the figure as a 2D array of cells -/
def Figure := Array (Array Cell)

/-- Checks if a figure has alternating black and white cells -/
def hasAlternatingColors (fig : Figure) : Prop := sorry

/-- Checks if the middle diagonal of a figure is black -/
def hasBlackDiagonal (fig : Figure) : Prop := sorry

/-- Counts the number of black cells in a figure -/
def countBlackCells (fig : Figure) : Nat := sorry

/-- Represents a 1x2 rectangle placement in the figure -/
structure Rectangle where
  row : Nat
  col : Nat

/-- Checks if a rectangle placement is valid (spans one black and one white cell) -/
def isValidRectangle (fig : Figure) (rect : Rectangle) : Prop := sorry

/-- The main theorem -/
theorem max_rectangles (fig : Figure) 
  (h1 : hasAlternatingColors fig)
  (h2 : hasBlackDiagonal fig) :
  (∃ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) ∧ 
    rects.length = countBlackCells fig) ∧
  (∀ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) → 
    rects.length ≤ countBlackCells fig) := by
  sorry

end max_rectangles_l2180_218001


namespace maximum_marks_l2180_218032

theorem maximum_marks (victor_marks : ℕ) (max_marks : ℕ) (h1 : victor_marks = 368) (h2 : 92 * max_marks = 100 * victor_marks) : max_marks = 400 := by
  sorry

end maximum_marks_l2180_218032


namespace bug_position_after_2023_jumps_l2180_218071

/-- Represents the points on the circle -/
inductive Point : Type
| one | two | three | four | five | six | seven

/-- The next point function, implementing the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.six
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs n jumps starting from a given point -/
def jumpN (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

/-- The main theorem stating that after 2023 jumps from point 7, the bug lands on point 2 -/
theorem bug_position_after_2023_jumps :
  jumpN Point.seven 2023 = Point.two := by sorry

end bug_position_after_2023_jumps_l2180_218071


namespace triangle_angle_measure_l2180_218029

theorem triangle_angle_measure (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 34 →
  angle2 = 53 →
  angle3 = 27 →
  angle1 + angle2 + angle3 + angle4 = 180 →
  angle4 = 114 := by
sorry

end triangle_angle_measure_l2180_218029


namespace circle_center_sum_l2180_218069

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x - 4*y + 14

/-- The center of a circle given by its equation -/
def CircleCenter (x y : ℝ) : Prop :=
  CircleEquation x y ∧ ∀ a b : ℝ, CircleEquation a b → (a - x)^2 + (b - y)^2 ≤ (x - x)^2 + (y - y)^2

theorem circle_center_sum :
  ∀ x y : ℝ, CircleCenter x y → x + y = 3 := by sorry

end circle_center_sum_l2180_218069


namespace solution_volume_l2180_218015

theorem solution_volume (V : ℝ) : 
  (0.20 * V + 0.60 * 4 = 0.36 * (V + 4)) → V = 6 := by
  sorry

end solution_volume_l2180_218015


namespace degree_of_product_polynomial_l2180_218011

/-- The degree of a polynomial (x^2+1)^5 * (x^3+1)^2 * (x+1)^3 -/
theorem degree_of_product_polynomial : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 1)^5 * (X^3 + 1)^2 * (X + 1)^3 ∧ 
  Polynomial.degree p = 19 := by
  sorry

end degree_of_product_polynomial_l2180_218011


namespace product_of_square_roots_l2180_218054

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (8 * q^5) = 20 * q^4 * Real.sqrt (3 * q) :=
by sorry

end product_of_square_roots_l2180_218054


namespace horner_method_v₃_l2180_218081

def f (x : ℝ) : ℝ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

def horner_v₀ : ℝ := 7
def horner_v₁ (x : ℝ) : ℝ := horner_v₀ * x + 5
def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 3
def horner_v₃ (x : ℝ) : ℝ := horner_v₂ x * x + 1

theorem horner_method_v₃ : horner_v₃ 2 = 83 :=
sorry

end horner_method_v₃_l2180_218081


namespace geometric_sequence_property_l2180_218003

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 10 = -2) :
  a 4 * a 7 = -2 := by
sorry

end geometric_sequence_property_l2180_218003


namespace grid_cut_into_L_shapes_l2180_218057

/-- An L-shaped piece is a shape formed by three squares in an L configuration -/
def LShape : Type := Unit

/-- A grid is a collection of squares arranged in rows and columns -/
def Grid (m n : ℕ) : Type := Fin m → Fin n → Bool

/-- A function that checks if a grid can be cut into L-shaped pieces -/
def can_be_cut_into_L_shapes (g : Grid m n) : Prop := sorry

/-- Main theorem: Any (3n+1) × (3n+1) grid with one square removed can be cut into L-shaped pieces -/
theorem grid_cut_into_L_shapes (n : ℕ) (h : n > 0) :
  ∀ (g : Grid (3*n+1) (3*n+1)), (∃ (i j : Fin (3*n+1)), ¬g i j) →
  can_be_cut_into_L_shapes g :=
sorry

end grid_cut_into_L_shapes_l2180_218057


namespace trust_meteorologist_l2180_218086

-- Define the probability of a clear day
def prob_clear_day : ℝ := 0.74

-- Define the accuracy of a senator's forecast (as a variable)
variable (p : ℝ)

-- Define the accuracy of the meteorologist's forecast
def meteorologist_accuracy (p : ℝ) : ℝ := 1.5 * p

-- Define the event of both senators predicting a clear day and the meteorologist predicting rain
def forecast_event (p : ℝ) : ℝ := 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day + 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day)

-- Theorem statement
theorem trust_meteorologist (p : ℝ) (h1 : 0 < p) (h2 : p < 1) : 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day) / forecast_event p > 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day / forecast_event p :=
sorry

end trust_meteorologist_l2180_218086


namespace surface_area_increase_l2180_218027

/-- The increase in surface area when a cube of edge length a is cut into 27 congruent smaller cubes -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  let original_surface_area := 6 * a^2
  let small_cube_edge := a / 3
  let small_cube_surface_area := 6 * small_cube_edge^2
  let total_new_surface_area := 27 * small_cube_surface_area
  total_new_surface_area - original_surface_area = 12 * a^2 := by
sorry


end surface_area_increase_l2180_218027


namespace subway_speed_difference_l2180_218052

/-- The speed function of the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the existence of the time when the train was 28 km/h slower -/
theorem subway_speed_difference :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ 
  speed 7 - speed t = 28 ∧ 
  t = 5 := by sorry

end subway_speed_difference_l2180_218052


namespace coefficient_theorem_l2180_218058

def expression (x : ℝ) : ℝ := 2 * (3 * x - 5) + 5 * (6 - 3 * x^2 + 2 * x) - 9 * (4 * x - 2)

theorem coefficient_theorem :
  ∃ (a b c : ℝ), ∀ x, expression x = a * x^2 + b * x + c ∧ a = -15 ∧ b = -20 := by
  sorry

end coefficient_theorem_l2180_218058


namespace remainder_7459_div_9_l2180_218030

theorem remainder_7459_div_9 : 
  7459 % 9 = (7 + 4 + 5 + 9) % 9 := by sorry

end remainder_7459_div_9_l2180_218030


namespace x_value_l2180_218055

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := 2 * a - b

-- Theorem statement
theorem x_value :
  ∃ x : ℚ, triangle x (triangle 1 3) = 2 ∧ x = 1/2 :=
by
  sorry

end x_value_l2180_218055


namespace equivalence_conditions_l2180_218025

theorem equivalence_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x < y) ↔ (1 / x > 1 / y) ∧ (x - y < Real.cos x - Real.cos y) ∧ (Real.exp x - Real.exp y < x^2 - y^2) := by
  sorry

end equivalence_conditions_l2180_218025


namespace negation_of_existence_negation_of_inequality_l2180_218017

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, x^2 > 2^x) ↔ (∀ x : ℝ, x^2 ≤ 2^x) := by sorry

end negation_of_existence_negation_of_inequality_l2180_218017


namespace white_closed_under_add_mul_l2180_218046

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A function that assigns a color to each positive integer --/
def coloring : ℕ+ → Color := sorry

/-- The property that the sum of two differently colored numbers is black --/
axiom sum_diff_color_black :
  ∀ (a b : ℕ+), coloring a ≠ coloring b → coloring (a + b) = Color.Black

/-- The property that there are infinitely many white numbers --/
axiom infinitely_many_white :
  ∀ (n : ℕ), ∃ (m : ℕ+), m > n ∧ coloring m = Color.White

/-- The theorem stating that the set of white numbers is closed under addition and multiplication --/
theorem white_closed_under_add_mul :
  ∀ (a b : ℕ+),
    coloring a = Color.White →
    coloring b = Color.White →
    coloring (a + b) = Color.White ∧ coloring (a * b) = Color.White :=
by sorry

end white_closed_under_add_mul_l2180_218046


namespace catch_up_theorem_l2180_218006

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

/-- The speed of the first tourist on bicycle in km/h -/
def speed_bicycle : ℝ := 16

/-- The speed of the second tourist on motorcycle in km/h -/
def speed_motorcycle : ℝ := 56

/-- The initial travel time of the first tourist before the break in hours -/
def initial_travel_time : ℝ := 1.5

/-- The break time of the first tourist in hours -/
def break_time : ℝ := 1.5

/-- The time delay between the start of the first and second tourist in hours -/
def start_delay : ℝ := 4

theorem catch_up_theorem :
  ∃ t : ℝ, t > 0 ∧
  speed_bicycle * (initial_travel_time + t) = 
  speed_motorcycle * t ∧
  catch_up_distance = speed_motorcycle * t :=
sorry

end catch_up_theorem_l2180_218006


namespace students_walking_to_school_l2180_218041

theorem students_walking_to_school 
  (total_students : ℕ) 
  (walking_minus_public : ℕ) 
  (h1 : total_students = 41)
  (h2 : walking_minus_public = 3) :
  let walking := (total_students + walking_minus_public) / 2
  walking = 22 := by
  sorry

end students_walking_to_school_l2180_218041


namespace train_length_l2180_218043

/-- The length of a train given its speed and time to pass a stationary observer -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 144 → time = 4 → speed * time * (1000 / 3600) = 160 := by
  sorry

end train_length_l2180_218043


namespace span_equality_iff_multiple_l2180_218021

theorem span_equality_iff_multiple (α₁ β₁ γ₁ α₂ β₂ γ₂ : ℝ) 
  (h₁ : α₁ + β₁ + γ₁ ≠ 0) (h₂ : α₂ + β₂ + γ₂ ≠ 0) :
  Submodule.span ℝ {(α₁, β₁, γ₁)} = Submodule.span ℝ {(α₂, β₂, γ₂)} ↔ 
  ∃ (k : ℝ), k ≠ 0 ∧ (α₁, β₁, γ₁) = (k * α₂, k * β₂, k * γ₂) :=
by sorry

end span_equality_iff_multiple_l2180_218021


namespace workday_percentage_theorem_l2180_218040

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the duration of the break in minutes -/
def break_duration : ℕ := 30

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_and_break_time : ℕ :=
  first_meeting_duration + second_meeting_duration + break_duration

/-- Theorem: The percentage of the workday spent in meetings or on break is 35% -/
theorem workday_percentage_theorem :
  (total_meeting_and_break_time : ℚ) / (workday_duration : ℚ) * 100 = 35 := by
  sorry

end workday_percentage_theorem_l2180_218040


namespace schedule_arrangements_eq_192_l2180_218045

/-- The number of ways to arrange 6 distinct lessons into 6 time slots -/
def schedule_arrangements (total_lessons : ℕ) (morning_slots : ℕ) (afternoon_slots : ℕ) 
  (morning_constraint : ℕ) (afternoon_constraint : ℕ) : ℕ := 
  (morning_slots.choose morning_constraint) * 
  (afternoon_slots.choose afternoon_constraint) * 
  (Nat.factorial (total_lessons - morning_constraint - afternoon_constraint))

/-- Theorem stating that the number of schedule arrangements is 192 -/
theorem schedule_arrangements_eq_192 : 
  schedule_arrangements 6 4 2 1 1 = 192 := by
  sorry

end schedule_arrangements_eq_192_l2180_218045


namespace younger_brother_age_l2180_218068

theorem younger_brother_age (x y : ℕ) 
  (h1 : x + y = 46) 
  (h2 : y = x / 3 + 10) : 
  y = 19 := by
  sorry

end younger_brother_age_l2180_218068


namespace birthday_spending_l2180_218002

theorem birthday_spending (initial_amount remaining_amount : ℕ) : 
  initial_amount = 7 → remaining_amount = 5 → initial_amount - remaining_amount = 2 := by
  sorry

end birthday_spending_l2180_218002


namespace sally_quarters_count_l2180_218091

/-- Given that Sally had 760 quarters initially and received 418 more quarters,
    prove that she now has 1178 quarters in total. -/
theorem sally_quarters_count (initial : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : initial = 760)
    (h2 : additional = 418)
    (h3 : total = initial + additional) :
  total = 1178 := by
  sorry

end sally_quarters_count_l2180_218091


namespace linear_function_quadrants_l2180_218093

/-- A linear function passing through the first quadrant -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y > 0

/-- A linear function passing through the fourth quadrant -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y < 0

/-- Theorem stating that a linear function y = kx + b with kb < 0 passes through both
    the first and fourth quadrants -/
theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end linear_function_quadrants_l2180_218093


namespace right_angle_point_coordinates_l2180_218004

/-- Given points A, B, and P, where P is on the y-axis and forms a right angle with AB, 
    prove that P has coordinates (0, -11) -/
theorem right_angle_point_coordinates 
  (A B P : ℝ × ℝ)
  (hA : A = (-3, -2))
  (hB : B = (6, 1))
  (hP_y_axis : P.1 = 0)
  (h_right_angle : (P.2 - A.2) * (B.2 - A.2) = -(P.1 - A.1) * (B.1 - A.1)) :
  P = (0, -11) := by
  sorry

end right_angle_point_coordinates_l2180_218004


namespace quadratic_roots_sum_squares_l2180_218079

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 20) → 
  |h| = 1 := by
sorry

end quadratic_roots_sum_squares_l2180_218079


namespace matchmaking_theorem_l2180_218070

-- Define a bipartite graph
def BipartiteGraph (α : Type) := (α → Bool) → α → α → Prop

-- Define a matching in a bipartite graph
def Matching (α : Type) (G : BipartiteGraph α) (M : α → α → Prop) :=
  ∀ x y z, M x y → M x z → y = z

-- Define a perfect matching for a subset
def PerfectMatchingForSubset (α : Type) (G : BipartiteGraph α) (S : Set α) (M : α → α → Prop) :=
  Matching α G M ∧ ∀ x ∈ S, ∃ y, M x y

-- Main theorem
theorem matchmaking_theorem (α : Type) (G : BipartiteGraph α) 
  (B W : Set α) (B1 : Set α) (W2 : Set α) 
  (hB1 : B1 ⊆ B) (hW2 : W2 ⊆ W)
  (M1 : α → α → Prop) (M2 : α → α → Prop)
  (hM1 : PerfectMatchingForSubset α G B1 M1)
  (hM2 : PerfectMatchingForSubset α G W2 M2) :
  ∃ M : α → α → Prop, 
    Matching α G M ∧ 
    (∀ x y, M1 x y → M x y) ∧ 
    (∀ x y, M2 x y → M x y) :=
sorry

end matchmaking_theorem_l2180_218070


namespace price_reduction_l2180_218012

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 ∧ first_reduction < 100 →
  (1 - first_reduction / 100) * (1 - 0.3) = (1 - 0.475) →
  first_reduction = 25 := by
  sorry

end price_reduction_l2180_218012


namespace exactly_three_valid_pairs_l2180_218020

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 - (360 / n)

/-- Predicate for valid pairs of regular polygons -/
def valid_pair (k r : ℕ) : Prop :=
  k > 2 ∧ r > 2 ∧ (interior_angle r) / (interior_angle k) = 4 / 3

/-- The number of valid pairs of regular polygons -/
def num_valid_pairs : ℕ := 3

/-- Theorem stating that there are exactly 3 valid pairs -/
theorem exactly_three_valid_pairs :
  ∃! (s : Finset (ℕ × ℕ)), s.card = num_valid_pairs ∧ 
  (∀ (k r : ℕ), (k, r) ∈ s ↔ valid_pair k r) :=
sorry

end exactly_three_valid_pairs_l2180_218020


namespace composition_equality_l2180_218072

/-- Given two functions f and g, prove that their composition at x = 3 equals 103 -/
theorem composition_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 3) 
  (hg : ∀ x, g x = (x + 2) ^ 2) : 
  f (g 3) = 103 := by
  sorry

end composition_equality_l2180_218072


namespace five_digit_multiple_of_nine_l2180_218038

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ 56170 + d ≡ 0 [MOD 9] := by
  sorry

end five_digit_multiple_of_nine_l2180_218038


namespace iris_count_after_addition_l2180_218034

/-- Calculates the number of irises needed to maintain a ratio of 3:7 with roses -/
def calculate_irises (initial_roses : ℕ) (added_roses : ℕ) : ℕ :=
  let total_roses := initial_roses + added_roses
  let irises := (3 * total_roses) / 7
  irises

theorem iris_count_after_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (h1 : initial_roses = 35) 
  (h2 : added_roses = 25) : 
  calculate_irises initial_roses added_roses = 25 := by
sorry

#eval calculate_irises 35 25

end iris_count_after_addition_l2180_218034


namespace tangent_line_intersection_l2180_218035

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -1 ∧
  (f a x₁ = (a + 1) * x₁) ∧
  (f a x₂ = (a + 1) * x₂) ∧
  (∀ x : ℝ, f a x = (a + 1) * x → x = x₁ ∨ x = x₂) :=
sorry

end tangent_line_intersection_l2180_218035


namespace employee_age_when_hired_l2180_218063

/-- The age of an employee when hired, given the rule of 70 and years worked until retirement eligibility -/
def age_when_hired (years_worked : ℕ) : ℕ :=
  70 - years_worked

theorem employee_age_when_hired :
  age_when_hired 19 = 51 := by
  sorry

end employee_age_when_hired_l2180_218063


namespace divisible_by_eight_l2180_218022

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 6 * n^2 + 4 * n + (-1)^n * 9 + 7 = 8 * k := by
  sorry

end divisible_by_eight_l2180_218022


namespace time_to_park_l2180_218019

/-- Represents the jogging scenario with constant pace -/
structure JoggingScenario where
  pace : ℝ  -- Jogging pace in minutes per mile
  cafe_distance : ℝ  -- Distance to café in miles
  cafe_time : ℝ  -- Time to jog to café in minutes
  park_distance : ℝ  -- Distance to park in miles

/-- Given a jogging scenario with constant pace, proves that the time to jog to the park is 36 minutes -/
theorem time_to_park (scenario : JoggingScenario)
  (h1 : scenario.cafe_distance = 3)
  (h2 : scenario.cafe_time = 24)
  (h3 : scenario.park_distance = 4.5)
  (h4 : scenario.pace > 0) :
  scenario.pace * scenario.park_distance = 36 := by
  sorry

#check time_to_park

end time_to_park_l2180_218019


namespace eight_chairs_bought_l2180_218037

/-- Represents the chair purchase scenario at Big Lots --/
structure ChairPurchase where
  normalPrice : ℝ
  initialDiscount : ℝ
  additionalDiscount : ℝ
  totalCost : ℝ
  minChairsForAdditionalDiscount : ℕ

/-- Calculates the number of chairs bought given the purchase conditions --/
def calculateChairsBought (purchase : ChairPurchase) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 8 chairs were bought --/
theorem eight_chairs_bought : 
  let purchase : ChairPurchase := {
    normalPrice := 20,
    initialDiscount := 0.25,
    additionalDiscount := 1/3,
    totalCost := 105,
    minChairsForAdditionalDiscount := 5
  }
  calculateChairsBought purchase = 8 := by
  sorry

end eight_chairs_bought_l2180_218037


namespace sum_of_cubes_minus_product_l2180_218080

theorem sum_of_cubes_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 400 := by
sorry

end sum_of_cubes_minus_product_l2180_218080


namespace divide_by_fraction_twelve_divided_by_one_sixth_l2180_218090

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6 : ℚ) = 72 := by sorry

end divide_by_fraction_twelve_divided_by_one_sixth_l2180_218090


namespace abs_neg_two_thirds_eq_two_thirds_l2180_218024

theorem abs_neg_two_thirds_eq_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end abs_neg_two_thirds_eq_two_thirds_l2180_218024


namespace relay_race_distance_l2180_218087

/-- Proves that in a 5-member relay team where one member runs twice the distance of others,
    and the total race distance is 18 km, each of the other members runs 3 km. -/
theorem relay_race_distance (team_size : ℕ) (ralph_multiplier : ℕ) (total_distance : ℝ) :
  team_size = 5 →
  ralph_multiplier = 2 →
  total_distance = 18 →
  ∃ (other_distance : ℝ),
    other_distance = 3 ∧
    (team_size - 1) * other_distance + ralph_multiplier * other_distance = total_distance :=
by sorry

end relay_race_distance_l2180_218087


namespace quadratic_shift_sum_l2180_218042

/-- Given a quadratic function f(x) = 3x^2 + 2x + 4, when shifted 3 units to the left,
    it becomes g(x) = a*x^2 + b*x + c. This theorem proves that a + b + c = 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 3*(x+3)^2 + 2*(x+3) + 4 = a*x^2 + b*x + c) → 
  a + b + c = 60 := by
sorry


end quadratic_shift_sum_l2180_218042


namespace distance_between_points_l2180_218084

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -1)
  let p2 : ℝ × ℝ := (7, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 74 := by
  sorry

#check distance_between_points

end distance_between_points_l2180_218084


namespace concentric_circles_shaded_area_l2180_218016

/-- Given two concentric circles where the smaller circle's radius is half of the larger circle's radius,
    and the area of the larger circle is 144π, the sum of the areas of the upper halves of both circles
    is equal to 90π. -/
theorem concentric_circles_shaded_area (R r : ℝ) : 
  R > 0 ∧ r = R / 2 ∧ π * R^2 = 144 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 90 * π := by
  sorry


end concentric_circles_shaded_area_l2180_218016


namespace salary_increase_percentage_l2180_218083

theorem salary_increase_percentage
  (total_employees : ℕ)
  (travel_allowance_percentage : ℚ)
  (no_increase_count : ℕ)
  (h1 : total_employees = 480)
  (h2 : travel_allowance_percentage = 1/5)
  (h3 : no_increase_count = 336) :
  (total_employees - no_increase_count - (travel_allowance_percentage * total_employees)) / total_employees = 1/10 := by
  sorry

end salary_increase_percentage_l2180_218083
