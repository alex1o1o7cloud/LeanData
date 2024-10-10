import Mathlib

namespace double_elim_advantage_l3569_356986

-- Define the probability of team A winning against other teams
variable (p : ℝ)

-- Define the conditions
def knockout_prob := p^2
def double_elim_prob := p^3 * (3 - 2*p)

-- State the theorem
theorem double_elim_advantage (h1 : 1/2 < p) (h2 : p < 1) :
  knockout_prob p < double_elim_prob p :=
sorry

end double_elim_advantage_l3569_356986


namespace subcommittee_count_l3569_356963

def planning_committee_size : ℕ := 12
def teachers_in_committee : ℕ := 5
def subcommittee_size : ℕ := 5
def min_teachers_in_subcommittee : ℕ := 2

theorem subcommittee_count : 
  (Finset.sum (Finset.range (teachers_in_committee - min_teachers_in_subcommittee + 1))
    (fun k => Nat.choose teachers_in_committee (k + min_teachers_in_subcommittee) * 
              Nat.choose (planning_committee_size - teachers_in_committee) (subcommittee_size - (k + min_teachers_in_subcommittee)))) = 596 := by
  sorry

end subcommittee_count_l3569_356963


namespace other_number_proof_l3569_356974

theorem other_number_proof (a b : ℕ) (h1 : a + b = 62) (h2 : a = 27) : b = 35 := by
  sorry

end other_number_proof_l3569_356974


namespace remainder_difference_l3569_356994

theorem remainder_difference (d r : ℕ) : 
  d > 1 → 
  2023 % d = r → 
  2459 % d = r → 
  3571 % d = r → 
  d - r = 1 := by
sorry

end remainder_difference_l3569_356994


namespace sequence_property_l3569_356939

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, sequence_sum a n + a n = 4 - 1 / (2 ^ (n.val - 2))) →
  (∀ n : ℕ+, a n = n.val / (2 ^ (n.val - 1))) :=
sorry

end sequence_property_l3569_356939


namespace decimal_point_removal_l3569_356925

theorem decimal_point_removal (x y z : ℝ) (hx : x = 1.6) (hy : y = 16) (hz : z = 14.4) :
  y - x = z := by sorry

end decimal_point_removal_l3569_356925


namespace west_7m_is_negative_7m_l3569_356928

/-- Represents the direction of movement on an east-west road -/
inductive Direction
  | East
  | West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_7m_is_negative_7m
  (h : Movement.toSigned { direction := Direction.East, distance := 3 } = 3) :
  Movement.toSigned { direction := Direction.West, distance := 7 } = -7 := by
  sorry

end west_7m_is_negative_7m_l3569_356928


namespace indeterminate_relation_product_and_means_l3569_356973

/-- Given two positive real numbers, their arithmetic mean, and their geometric mean,
    the relationship between the product of the numbers and the product of their means
    cannot be determined. -/
theorem indeterminate_relation_product_and_means (a b : ℝ) (A G : ℝ) 
    (ha : 0 < a) (hb : 0 < b)
    (hA : A = (a + b) / 2)
    (hG : G = Real.sqrt (a * b)) :
    ¬ ∀ (R : ℝ → ℝ → Prop), R (a * b) (A * G) ∨ R (A * G) (a * b) := by
  sorry

end indeterminate_relation_product_and_means_l3569_356973


namespace burgers_spent_l3569_356938

def total_allowance : ℚ := 50

def movies_fraction : ℚ := 1/4
def music_fraction : ℚ := 3/10
def ice_cream_fraction : ℚ := 2/5

def burgers_amount : ℚ := total_allowance - (movies_fraction * total_allowance + music_fraction * total_allowance + ice_cream_fraction * total_allowance)

theorem burgers_spent :
  burgers_amount = 5/2 := by sorry

end burgers_spent_l3569_356938


namespace p_sufficient_not_necessary_for_q_l3569_356985

/-- A function that represents the cubic polynomial f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: ∀x ∈ ℝ, x²-4x+3m > 0 -/
def condition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0

/-- The condition q: f(x) is strictly increasing on (-∞,+∞) -/
def condition_q (m : ℝ) : Prop := StrictMono (f m)

/-- Theorem stating that p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m → condition_q m) ∧
  (∃ m : ℝ, condition_q m ∧ ¬condition_p m) :=
sorry

end p_sufficient_not_necessary_for_q_l3569_356985


namespace central_angles_sum_l3569_356968

theorem central_angles_sum (y : ℝ) : 
  (6 * y + 7 * y + 3 * y + y) * (π / 180) = 2 * π → y = 360 / 17 := by
sorry

end central_angles_sum_l3569_356968


namespace f_properties_l3569_356983

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * Real.pi - x) - Real.cos (Real.pi / 2 + x) + 1

theorem f_properties :
  (∀ x, -1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ k : ℤ, ∀ x, -5 * Real.pi / 6 + 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + 2 * k * Real.pi → 
    ∀ y, x ≤ y → f x ≤ f y) ∧
  (∀ α, f α = 13 / 5 → Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 → 
    Real.cos (2 * α) = (7 - 24 * Real.sqrt 3) / 50) :=
by sorry

end f_properties_l3569_356983


namespace second_number_calculation_l3569_356916

theorem second_number_calculation (a b : ℝ) (h1 : a = 1600) (h2 : 0.20 * a = 0.20 * b + 190) : b = 650 := by
  sorry

end second_number_calculation_l3569_356916


namespace total_cost_is_correct_l3569_356918

def dress_shirt_price : ℝ := 25
def pants_price : ℝ := 35
def socks_price : ℝ := 10
def dress_shirt_quantity : ℕ := 4
def pants_quantity : ℕ := 2
def socks_quantity : ℕ := 3
def dress_shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.20
def socks_discount : ℝ := 0.10
def tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 12.50

def total_cost : ℝ :=
  let dress_shirts_total := dress_shirt_price * dress_shirt_quantity * (1 - dress_shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let socks_total := socks_price * socks_quantity * (1 - socks_discount)
  let subtotal := dress_shirts_total + pants_total + socks_total
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 197.30 := by
  sorry

end total_cost_is_correct_l3569_356918


namespace forty_percent_value_l3569_356952

theorem forty_percent_value (x : ℝ) (h : 0.1 * x = 40) : 0.4 * x = 160 := by
  sorry

end forty_percent_value_l3569_356952


namespace max_catered_children_correct_l3569_356920

structure MealData where
  total_adults : ℕ
  total_children : ℕ
  prepared_veg_adults : ℕ
  prepared_nonveg_adults : ℕ
  prepared_vegan_adults : ℕ
  prepared_veg_children : ℕ
  prepared_nonveg_children : ℕ
  prepared_vegan_children : ℕ
  pref_veg_adults : ℕ
  pref_nonveg_adults : ℕ
  pref_vegan_adults : ℕ
  pref_veg_children : ℕ
  pref_nonveg_children : ℕ
  pref_vegan_children : ℕ
  eaten_veg_adults : ℕ
  eaten_nonveg_adults : ℕ
  eaten_vegan_adults : ℕ

def max_catered_children (data : MealData) : ℕ × ℕ × ℕ :=
  let remaining_veg := data.prepared_veg_adults + data.prepared_veg_children - data.eaten_veg_adults
  let remaining_nonveg := data.prepared_nonveg_adults + data.prepared_nonveg_children - data.eaten_nonveg_adults
  let remaining_vegan := data.prepared_vegan_adults + data.prepared_vegan_children - data.eaten_vegan_adults
  (min remaining_veg data.pref_veg_children,
   min remaining_nonveg data.pref_nonveg_children,
   min remaining_vegan data.pref_vegan_children)

theorem max_catered_children_correct (data : MealData) : 
  data.total_adults = 80 ∧
  data.total_children = 120 ∧
  data.prepared_veg_adults = 70 ∧
  data.prepared_nonveg_adults = 75 ∧
  data.prepared_vegan_adults = 5 ∧
  data.prepared_veg_children = 90 ∧
  data.prepared_nonveg_children = 25 ∧
  data.prepared_vegan_children = 5 ∧
  data.pref_veg_adults = 45 ∧
  data.pref_nonveg_adults = 30 ∧
  data.pref_vegan_adults = 5 ∧
  data.pref_veg_children = 100 ∧
  data.pref_nonveg_children = 15 ∧
  data.pref_vegan_children = 5 ∧
  data.eaten_veg_adults = 42 ∧
  data.eaten_nonveg_adults = 25 ∧
  data.eaten_vegan_adults = 5
  →
  max_catered_children data = (100, 15, 5) := by
sorry

end max_catered_children_correct_l3569_356920


namespace tax_free_items_cost_l3569_356979

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08) : 
  total_paid - sales_tax / tax_rate = 14 :=
by
  sorry

end tax_free_items_cost_l3569_356979


namespace vector_rotation_angle_l3569_356972

theorem vector_rotation_angle (p : ℂ) (α : ℝ) (h_p : p ≠ 0) :
  p + p * Complex.exp (2 * α * Complex.I) = p * Complex.exp (α * Complex.I) →
  α = π / 3 + 2 * π * ↑k ∨ α = -π / 3 + 2 * π * ↑n :=
by sorry

end vector_rotation_angle_l3569_356972


namespace roots_of_unity_count_l3569_356935

theorem roots_of_unity_count (a b c : ℤ) : 
  ∃ (roots : Finset ℂ), 
    (∀ z ∈ roots, z^3 = 1 ∧ z^3 + a*z^2 + b*z + c = 0) ∧ 
    Finset.card roots = 3 :=
sorry

end roots_of_unity_count_l3569_356935


namespace inequality_system_integer_solutions_l3569_356905

def inequality_system (x : ℝ) : Prop :=
  (3*x - 2)/3 ≥ 1 ∧ 3*x + 5 > 4*x - 2

def integer_solutions : Set ℤ := {2, 3, 4, 5, 6}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), n ∈ integer_solutions ↔ inequality_system (n : ℝ) :=
sorry

end inequality_system_integer_solutions_l3569_356905


namespace original_count_pingpong_shuttlecock_l3569_356926

theorem original_count_pingpong_shuttlecock : ∀ (n : ℕ),
  (∃ (x : ℕ), n = 5 * x ∧ n = 3 * x + 16) →
  n = 40 := by
  sorry

end original_count_pingpong_shuttlecock_l3569_356926


namespace egg_cost_calculation_l3569_356903

def dozen : ℕ := 12

theorem egg_cost_calculation (total_cost : ℚ) (num_dozens : ℕ) 
  (h1 : total_cost = 18) 
  (h2 : num_dozens = 3) : 
  total_cost / (num_dozens * dozen) = 1/2 := by
  sorry

end egg_cost_calculation_l3569_356903


namespace repeated_root_implies_m_equals_two_l3569_356953

/-- Given that the equation (m-1)/(x-1) - x/(x-1) = 0 has a repeated root, prove that m = 2 -/
theorem repeated_root_implies_m_equals_two (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (m - 1) / (x - 1) - x / (x - 1) = 0 ∧
   ∀ y : ℝ, y ≠ 1 → ((m - 1) / (y - 1) - y / (y - 1) = 0 → y = x)) →
  m = 2 :=
by sorry

end repeated_root_implies_m_equals_two_l3569_356953


namespace martin_fruits_l3569_356991

/-- Represents the number of fruits Martin initially had -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating -/
def remaining_oranges : ℕ := 50

/-- Represents the fraction of fruits Martin ate -/
def eaten_fraction : ℚ := 1/2

theorem martin_fruits :
  (initial_fruits : ℚ) * (1 - eaten_fraction) = remaining_oranges * 3 :=
sorry

end martin_fruits_l3569_356991


namespace simplify_and_evaluate_l3569_356933

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = 2023) :
  (x + y)^2 + (x + y)*(x - y) - 2*x^2 = 2*x*y := by
  sorry

end simplify_and_evaluate_l3569_356933


namespace inverse_as_linear_combination_l3569_356937

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, -1; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/10 ∧ d = 1/5 := by
  sorry

end inverse_as_linear_combination_l3569_356937


namespace largest_number_l3569_356969

theorem largest_number (x y z : ℝ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 82) (h4 : z - y = 10) (h5 : y - x = 4) :
  z = 106 / 3 := by
  sorry

end largest_number_l3569_356969


namespace no_cyclic_knight_tour_5x5_l3569_356956

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move --/
inductive KnightMove
  | move : Nat → Nat → KnightMove

/-- Represents a tour on the chessboard --/
structure Tour :=
  (moves : List KnightMove)
  (cyclic : Bool)

/-- Defines a valid knight's move --/
def isValidKnightMove (m : KnightMove) : Prop :=
  match m with
  | KnightMove.move x y => (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)

/-- Defines if a tour visits each square exactly once --/
def visitsEachSquareOnce (t : Tour) (b : Chessboard) : Prop :=
  t.moves.length = b.rows * b.cols

/-- Theorem: It's impossible for a knight to make a cyclic tour on a 5x5 chessboard
    visiting each square exactly once --/
theorem no_cyclic_knight_tour_5x5 :
  ∀ (t : Tour),
    t.cyclic →
    (∀ (m : KnightMove), m ∈ t.moves → isValidKnightMove m) →
    visitsEachSquareOnce t (Chessboard.mk 5 5) →
    False :=
sorry

end no_cyclic_knight_tour_5x5_l3569_356956


namespace necessary_but_not_sufficient_l3569_356941

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 + 5*x + 6 < 0 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x^2 + 5*x + 6 ≥ 0) :=
by sorry

end necessary_but_not_sufficient_l3569_356941


namespace find_p_l3569_356958

theorem find_p (m : ℕ) (p : ℕ) :
  m = 34 →
  ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ p)) →
  p = 35 := by
  sorry

end find_p_l3569_356958


namespace p_geq_q_l3569_356929

theorem p_geq_q (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := by
  sorry

end p_geq_q_l3569_356929


namespace shortest_chord_through_M_l3569_356975

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 10 = 0

-- Define point M
def point_M : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (l : ℝ × ℝ → Prop),
    (∀ x y, l (x, y) ↔ line_equation x y) →
    (l point_M) →
    (∀ other_line : ℝ × ℝ → Prop,
      (other_line point_M) →
      (∃ p, circle_equation p.1 p.2 ∧ other_line p) →
      (∃ p q : ℝ × ℝ, 
        p ≠ q ∧ 
        circle_equation p.1 p.2 ∧ circle_equation q.1 q.2 ∧ 
        l p ∧ l q ∧
        other_line p ∧ other_line q →
        (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ 
        (p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
  sorry

end shortest_chord_through_M_l3569_356975


namespace system_solution_l3569_356997

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 5 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 5 * y - 3 * z)
  (eq3 : x * z = 18 - 2 * x - 5 * z)
  (pos_x : x > 0) : x = 6 := by
  sorry

end system_solution_l3569_356997


namespace triangle_side_length_l3569_356965

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 120 * π / 180 →  -- Convert 120° to radians
  a = 2 * Real.sqrt 3 → 
  b = 2 → 
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) → 
  c = 2 := by sorry

end triangle_side_length_l3569_356965


namespace power_inequality_l3569_356976

theorem power_inequality (x : ℝ) (α : ℝ) (h1 : x > -1) :
  (0 < α ∧ α < 1 → (1 + x)^α ≤ 1 + α * x) ∧
  ((α < 0 ∨ α > 1) → (1 + x)^α ≥ 1 + α * x) := by
  sorry

end power_inequality_l3569_356976


namespace product_multiple_of_60_probability_l3569_356944

def is_multiple_of_60 (n : ℕ) : Prop := ∃ k : ℕ, n = 60 * k

def count_favorable_pairs : ℕ := 732

def total_pairs : ℕ := 60 * 60

theorem product_multiple_of_60_probability :
  (count_favorable_pairs : ℚ) / (total_pairs : ℚ) = 61 / 300 := by sorry

end product_multiple_of_60_probability_l3569_356944


namespace larry_jogging_time_l3569_356967

/-- Calculates the total jogging time in hours for two weeks given daily jogging time and days jogged each week -/
def total_jogging_time (daily_time : ℕ) (days_week1 : ℕ) (days_week2 : ℕ) : ℚ :=
  ((daily_time * days_week1 + daily_time * days_week2) : ℚ) / 60

/-- Theorem stating that Larry's total jogging time for two weeks is 4 hours -/
theorem larry_jogging_time :
  total_jogging_time 30 3 5 = 4 := by
  sorry

end larry_jogging_time_l3569_356967


namespace quadratic_roots_not_uniformly_increased_l3569_356927

theorem quadratic_roots_not_uniformly_increased (b c : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0) :
  ¬∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 + (b+1)*y1 + (c+1) = 0 ∧ 
    y2^2 + (b+1)*y2 + (c+1) = 0 ∧
    ∃ x1 x2 : ℝ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 ∧ y1 = x1 + 1 ∧ y2 = x2 + 1 :=
by sorry

end quadratic_roots_not_uniformly_increased_l3569_356927


namespace problem_solution_l3569_356934

theorem problem_solution (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + 2*c = 10) 
  (h3 : c = 4) : 
  a = 2 := by
sorry

end problem_solution_l3569_356934


namespace remainder_problem_l3569_356984

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 := by
  sorry

end remainder_problem_l3569_356984


namespace geometric_sequence_a10_l3569_356948

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a10 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 162) : 
  a 10 = 13122 := by
sorry

end geometric_sequence_a10_l3569_356948


namespace ellipse_y_axis_iff_m_greater_n_l3569_356978

/-- The equation of an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for m and n -/
def m_greater_n (m n : ℝ) : Prop :=
  m > n ∧ n > 0

theorem ellipse_y_axis_iff_m_greater_n (m n : ℝ) :
  is_ellipse_y_axis m n ↔ m_greater_n m n :=
sorry

end ellipse_y_axis_iff_m_greater_n_l3569_356978


namespace no_equal_tuesdays_fridays_l3569_356970

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in a 30-day month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: No starting day results in equal Tuesdays and Fridays in a 30-day month -/
theorem no_equal_tuesdays_fridays :
  ∀ (startDay : DayOfWeek),
    countDayOccurrences startDay DayOfWeek.Tuesday ≠ 
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end no_equal_tuesdays_fridays_l3569_356970


namespace apples_bought_proof_l3569_356924

/-- The price of an orange in reals -/
def orange_price : ℝ := 2

/-- The price of an apple in reals -/
def apple_price : ℝ := 3

/-- An orange costs the same as half an apple plus half a real -/
axiom orange_price_relation : orange_price = apple_price / 2 + 1 / 2

/-- One-third of an apple costs the same as one-quarter of an orange plus half a real -/
axiom apple_price_relation : apple_price / 3 = orange_price / 4 + 1 / 2

/-- The number of apples that can be bought with the value of 5 oranges plus 5 reals -/
def apples_bought : ℕ := 5

theorem apples_bought_proof : 
  (5 * orange_price + 5) / apple_price = apples_bought := by sorry

end apples_bought_proof_l3569_356924


namespace unique_square_double_reverse_l3569_356922

theorem unique_square_double_reverse : ∃! x : ℕ,
  (10 ≤ x^2 ∧ x^2 < 100) ∧
  (10 ≤ 2*x ∧ 2*x < 100) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x^2 = 10*a + b ∧ 2*x = 10*b + a) ∧
  x^2 = 81 := by
  sorry

end unique_square_double_reverse_l3569_356922


namespace decimal_sum_l3569_356992

theorem decimal_sum : (0.08 : ℚ) + (0.003 : ℚ) + (0.0070 : ℚ) = (0.09 : ℚ) := by
  sorry

end decimal_sum_l3569_356992


namespace lance_workdays_per_week_l3569_356957

/-- Given Lance's work schedule and earnings, prove the number of workdays per week -/
theorem lance_workdays_per_week 
  (total_weekly_hours : ℕ) 
  (hourly_wage : ℚ) 
  (daily_earnings : ℚ) 
  (h1 : total_weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63)
  (h4 : ∃ (daily_hours : ℚ), daily_hours * hourly_wage = daily_earnings ∧ 
        daily_hours * (total_weekly_hours / daily_hours) = total_weekly_hours) :
  total_weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end lance_workdays_per_week_l3569_356957


namespace pirate_treasure_probability_l3569_356947

theorem pirate_treasure_probability :
  let n : ℕ := 5  -- number of islands
  let p_treasure : ℚ := 1/3  -- probability of treasure on an island
  let p_trap : ℚ := 1/6  -- probability of trap on an island
  let p_neither : ℚ := 1/2  -- probability of neither treasure nor trap on an island
  
  -- Probability of exactly 4 islands with treasure and 1 with neither
  (Nat.choose n 4 : ℚ) * p_treasure^4 * p_neither = 5/162 :=
by sorry

end pirate_treasure_probability_l3569_356947


namespace ratio_of_y_coordinates_l3569_356921

-- Define the ellipse
def Γ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Define the lines l₁ and l₂
def l₁ (x : ℝ) : Prop := x = -2
def l₂ (x : ℝ) : Prop := x = 2

-- Define the line l_CD
def l_CD (x : ℝ) : Prop := x = 1

-- Define the chords AB and CD (implicitly by their properties)
def chord_passes_through_P (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t) • P + t • B

-- Define points E and F
def E : ℝ × ℝ := (2, sorry)  -- y-coordinate to be determined
def F : ℝ × ℝ := (-2, sorry) -- y-coordinate to be determined

theorem ratio_of_y_coordinates :
  ∃ (A B C D : ℝ × ℝ),
    Γ A.1 A.2 ∧ Γ B.1 B.2 ∧ Γ C.1 C.2 ∧ Γ D.1 D.2 ∧
    chord_passes_through_P A B ∧ chord_passes_through_P C D ∧
    l_CD C.1 ∧ l_CD D.1 ∧
    (E.2 : ℝ) / (F.2 : ℝ) = -1/3 :=
sorry

end ratio_of_y_coordinates_l3569_356921


namespace largest_prime_divisor_factorial_sum_l3569_356950

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_prime_divisor_factorial_sum :
  ∃ (p : ℕ), isPrime p ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), isPrime q → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by sorry

end largest_prime_divisor_factorial_sum_l3569_356950


namespace curves_intersection_l3569_356995

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 4

/-- Intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(-1, -7), (3, 41)}

theorem curves_intersection :
  ∀ p : ℝ × ℝ, p ∈ intersection_points ↔ 
    (curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧
    ∀ x : ℝ, curve1 x = curve2 x → x = p.1 ∨ x = (if p.1 = -1 then 3 else -1) := by
  sorry

end curves_intersection_l3569_356995


namespace commission_calculation_l3569_356906

/-- The original commission held by the company for John --/
def original_commission : ℕ := sorry

/-- The advance agency fees taken by John --/
def advance_fees : ℕ := 8280

/-- The amount given to John by the accountant after one month --/
def amount_given : ℕ := 18500

/-- The incentive amount given to John --/
def incentive_amount : ℕ := 1780

/-- Theorem stating the relationship between the original commission and other amounts --/
theorem commission_calculation : 
  original_commission = amount_given + advance_fees - incentive_amount :=
by sorry

end commission_calculation_l3569_356906


namespace volvox_face_difference_l3569_356943

/-- A spherical polyhedron where each face has 5, 6, or 7 sides, and exactly three faces meet at each vertex. -/
structure VolvoxPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  f₅ : ℕ  -- number of pentagonal faces
  f₆ : ℕ  -- number of hexagonal faces
  f₇ : ℕ  -- number of heptagonal faces
  euler : V - E + F = 2
  face_sum : F = f₅ + f₆ + f₇
  edge_sum : 2 * E = 5 * f₅ + 6 * f₆ + 7 * f₇
  vertex_sum : 3 * V = 5 * f₅ + 6 * f₆ + 7 * f₇

/-- The number of pentagonal faces is always 12 more than the number of heptagonal faces. -/
theorem volvox_face_difference (p : VolvoxPolyhedron) : p.f₅ = p.f₇ + 12 := by
  sorry

end volvox_face_difference_l3569_356943


namespace hyperbola_C_tangent_intersection_product_l3569_356945

/-- Hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

/-- Point P on the line x = 2 -/
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

/-- Function to calculate mn given t -/
noncomputable def mn (t : ℝ) : ℝ := 6 * Real.sqrt 6 - 15

theorem hyperbola_C_tangent_intersection_product :
  hyperbola_C (-3) (Real.sqrt 6 / 2) →
  ∀ t : ℝ, ∃ m n : ℝ,
    (∃ A B : ℝ × ℝ, 
      hyperbola_C A.1 A.2 ∧ 
      hyperbola_C B.1 B.2 ∧ 
      -- PA and PB are tangent to C
      -- M and N are defined as in the problem
      mn t = m * n) :=
by sorry

end hyperbola_C_tangent_intersection_product_l3569_356945


namespace equation_solution_l3569_356981

theorem equation_solution : ∃ y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end equation_solution_l3569_356981


namespace complement_A_intersect_B_l3569_356907

-- Define the sets A and B
def A : Set ℝ := {x | 2^x ≤ 2 * Real.sqrt 2}
def B : Set ℝ := {x | Real.log (2 - x) < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Set.Ioo (3/2) 2 := by sorry

end complement_A_intersect_B_l3569_356907


namespace division_decomposition_l3569_356999

theorem division_decomposition : (36 : ℕ) / 3 = (30 / 3) + (6 / 3) := by sorry

end division_decomposition_l3569_356999


namespace books_read_l3569_356930

theorem books_read (total_books : ℕ) (total_movies : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  total_books = 10 →
  total_movies = 11 →
  movies_watched = 12 →
  books_read = (min movies_watched total_movies) + 1 →
  books_read = 12 := by
sorry

end books_read_l3569_356930


namespace leaves_per_sub_branch_l3569_356955

/-- Given a farm with trees, branches, and sub-branches, calculate the number of leaves per sub-branch. -/
theorem leaves_per_sub_branch 
  (num_trees : ℕ) 
  (branches_per_tree : ℕ) 
  (sub_branches_per_branch : ℕ) 
  (total_leaves : ℕ) 
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : total_leaves = 96000) :
  total_leaves / (num_trees * branches_per_tree * sub_branches_per_branch) = 60 := by
  sorry

#check leaves_per_sub_branch

end leaves_per_sub_branch_l3569_356955


namespace sequence_property_l3569_356987

theorem sequence_property (u : ℕ → ℤ) : 
  (∀ n m : ℕ, u (n * m) = u n + u m) → 
  (∀ n : ℕ, u n = 0) := by
sorry

end sequence_property_l3569_356987


namespace class_b_more_uniform_l3569_356914

/-- Represents a class of students participating in a gymnastics competition -/
structure GymClass where
  name : String
  num_students : Nat
  avg_height : Float
  height_variance : Float

/-- Determines which of two classes has more uniform heights based on their variances -/
def more_uniform_heights (class_a class_b : GymClass) : Prop :=
  class_a.height_variance < class_b.height_variance

/-- Theorem: Given the variances of Class A and Class B, Class B has more uniform heights -/
theorem class_b_more_uniform (class_a class_b : GymClass) 
  (h1 : class_a.name = "A" ∧ class_b.name = "B")
  (h2 : class_a.num_students = 18 ∧ class_b.num_students = 18)
  (h3 : class_a.avg_height = 1.72 ∧ class_b.avg_height = 1.72)
  (h4 : class_a.height_variance = 3.24)
  (h5 : class_b.height_variance = 1.63) :
  more_uniform_heights class_b class_a :=
by sorry

end class_b_more_uniform_l3569_356914


namespace sum_of_four_consecutive_integers_l3569_356971

theorem sum_of_four_consecutive_integers (S : ℤ) :
  (∃ n : ℤ, S = n + (n + 1) + (n + 2) + (n + 3)) ↔ (S - 6) % 4 = 0 := by
  sorry

end sum_of_four_consecutive_integers_l3569_356971


namespace marked_price_calculation_l3569_356904

theorem marked_price_calculation (total_price : ℝ) (discount_percentage : ℝ) : 
  total_price = 50 →
  discount_percentage = 60 →
  ∃ (marked_price : ℝ), 
    marked_price = 62.50 ∧ 
    2 * marked_price * (1 - discount_percentage / 100) = total_price :=
by sorry

end marked_price_calculation_l3569_356904


namespace curve_symmetry_l3569_356961

theorem curve_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a^2*x + (1-a^2)*y - 4 = 0 ↔ 
              y^2 + x^2 + a^2*y + (1-a^2)*x - 4 = 0) →
  a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by
sorry

end curve_symmetry_l3569_356961


namespace percentage_50_59_range_l3569_356998

/-- Represents the frequency distribution of scores in Mrs. Lopez's geometry class -/
structure ScoreDistribution :=
  (score_90_100 : Nat)
  (score_80_89 : Nat)
  (score_70_79 : Nat)
  (score_60_69 : Nat)
  (score_50_59 : Nat)
  (score_below_50 : Nat)

/-- Calculates the total number of students -/
def totalStudents (dist : ScoreDistribution) : Nat :=
  dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + 
  dist.score_60_69 + dist.score_50_59 + dist.score_below_50

/-- The actual score distribution in Mrs. Lopez's class -/
def lopezClassDist : ScoreDistribution :=
  { score_90_100 := 3
  , score_80_89 := 6
  , score_70_79 := 8
  , score_60_69 := 4
  , score_50_59 := 3
  , score_below_50 := 4
  }

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 3/28 * 100% -/
theorem percentage_50_59_range (dist : ScoreDistribution) :
  dist = lopezClassDist →
  (dist.score_50_59 : Rat) / (totalStudents dist : Rat) = 3 / 28 := by
  sorry

end percentage_50_59_range_l3569_356998


namespace triangle_proof_l3569_356910

/-- Given an acute triangle ABC with sides a and b, prove angle A and area. -/
theorem triangle_proof 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) -- Acute triangle condition
  (h_a : a = Real.sqrt 7) -- Side a
  (h_b : b = 3) -- Side b
  (h_sin_sum : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3) -- Given equation
  : A = π / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 := by
  sorry


end triangle_proof_l3569_356910


namespace farmer_milk_production_l3569_356980

/-- Calculates the total milk production for a farmer in a week -/
def total_milk_production (num_cows : ℕ) (milk_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  num_cows * milk_per_day * days_in_week

/-- Proves that a farmer with 52 cows, each producing 5 liters of milk per day,
    will get 1820 liters of milk in a week (7 days) -/
theorem farmer_milk_production :
  total_milk_production 52 5 7 = 1820 := by
  sorry

end farmer_milk_production_l3569_356980


namespace suitcase_theorem_l3569_356960

/-- Represents the suitcase scenario at the airport -/
structure SuitcaseScenario where
  total_suitcases : ℕ
  business_suitcases : ℕ
  placement_interval : ℕ

/-- The probability of businesspeople waiting exactly 2 minutes for their last suitcase -/
def exact_wait_probability (s : SuitcaseScenario) : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose s.total_suitcases s.business_suitcases)

/-- The expected waiting time for businesspeople's last suitcase in seconds -/
def expected_wait_time (s : SuitcaseScenario) : ℚ :=
  4020 / 11

/-- Theorem stating the probability and expected waiting time for the suitcase scenario -/
theorem suitcase_theorem (s : SuitcaseScenario) 
  (h1 : s.total_suitcases = 200)
  (h2 : s.business_suitcases = 10)
  (h3 : s.placement_interval = 2) :
  exact_wait_probability s = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10) ∧
  expected_wait_time s = 4020 / 11 := by
  sorry

#eval exact_wait_probability ⟨200, 10, 2⟩
#eval expected_wait_time ⟨200, 10, 2⟩

end suitcase_theorem_l3569_356960


namespace abc_maximum_l3569_356942

theorem abc_maximum (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + c = (a + c) * (b + c)) (h_sum : a + b + c = 2) :
  a * b * c ≤ 1 / 27 :=
sorry

end abc_maximum_l3569_356942


namespace negation_of_universal_positive_square_plus_one_l3569_356911

theorem negation_of_universal_positive_square_plus_one :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by sorry

end negation_of_universal_positive_square_plus_one_l3569_356911


namespace reflect_P_across_x_axis_l3569_356931

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of P(-3,2) reflected across the x-axis -/
theorem reflect_P_across_x_axis : 
  reflect_x (-3, 2) = (-3, -2) := by
  sorry

end reflect_P_across_x_axis_l3569_356931


namespace cherries_used_for_pie_l3569_356951

theorem cherries_used_for_pie (initial_cherries remaining_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : remaining_cherries = 17) :
  initial_cherries - remaining_cherries = 60 := by
sorry

end cherries_used_for_pie_l3569_356951


namespace cubic_sum_theorem_l3569_356940

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 9)/a = (b^3 + 9)/b ∧ (b^3 + 9)/b = (c^3 + 9)/c) :
  a^3 + b^3 + c^3 = -27 := by sorry

end cubic_sum_theorem_l3569_356940


namespace gcd_1443_999_l3569_356993

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by sorry

end gcd_1443_999_l3569_356993


namespace photo_arrangements_eq_24_l3569_356913

/-- The number of different arrangements for a teacher and two boys and two girls standing in a row,
    with the requirement that the two girls must stand together and the teacher cannot stand at either end. -/
def photo_arrangements : ℕ :=
  let n_people : ℕ := 5
  let n_boys : ℕ := 2
  let n_girls : ℕ := 2
  let n_teacher : ℕ := 1
  let girls_together : ℕ := 1  -- Treat the two girls as one unit
  let teacher_positions : ℕ := n_people - 2  -- Teacher can't be at either end
  Nat.factorial n_people / (Nat.factorial n_boys * Nat.factorial girls_together * Nat.factorial n_teacher)
    * teacher_positions

theorem photo_arrangements_eq_24 : photo_arrangements = 24 := by
  sorry

end photo_arrangements_eq_24_l3569_356913


namespace teaspoon_knife_ratio_l3569_356996

/-- Proves that the ratio of initial teaspoons to initial knives is 2:1 --/
theorem teaspoon_knife_ratio : 
  ∀ (initial_teaspoons : ℕ),
  let initial_knives : ℕ := 24
  let additional_knives : ℕ := initial_knives / 3
  let additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3
  let total_cutlery : ℕ := 112
  (initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery) →
  (initial_teaspoons : ℚ) / initial_knives = 2 := by
  sorry

end teaspoon_knife_ratio_l3569_356996


namespace tree_height_difference_l3569_356923

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let maple_height : ℚ := 75/4
  maple_height - pine_height = 13/2 := by sorry

end tree_height_difference_l3569_356923


namespace angelina_walking_speed_l3569_356932

/-- Angelina's walking problem -/
theorem angelina_walking_speed
  (distance_home_grocery : ℝ)
  (distance_grocery_gym : ℝ)
  (time_difference : ℝ)
  (h1 : distance_home_grocery = 960)
  (h2 : distance_grocery_gym = 480)
  (h3 : time_difference = 40)
  (h4 : distance_grocery_gym / (distance_home_grocery / speed_home_grocery) 
      = distance_grocery_gym / ((distance_home_grocery / speed_home_grocery) - time_difference))
  (h5 : speed_grocery_gym = 2 * speed_home_grocery) :
  speed_grocery_gym = 36 :=
by sorry

#check angelina_walking_speed

end angelina_walking_speed_l3569_356932


namespace red_marbles_in_bag_l3569_356962

theorem red_marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + 3)
  (h2 : (red_marbles : ℝ) / total_marbles * ((red_marbles - 1) : ℝ) / (total_marbles - 1) = 0.1) :
  red_marbles = 2 := by
sorry

end red_marbles_in_bag_l3569_356962


namespace distributive_property_division_l3569_356912

theorem distributive_property_division (a b c : ℝ) (hc : c ≠ 0) :
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := by
  sorry

end distributive_property_division_l3569_356912


namespace least_value_x_minus_y_plus_z_l3569_356982

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) 
  (h : (3 : ℕ) * x = (4 : ℕ) * y ∧ (4 : ℕ) * y = (7 : ℕ) * z) : 
  (∀ a b c : ℕ+, (3 : ℕ) * a = (4 : ℕ) * b ∧ (4 : ℕ) * b = (7 : ℕ) * c → 
    (x : ℤ) - (y : ℤ) + (z : ℤ) ≤ (a : ℤ) - (b : ℤ) + (c : ℤ)) ∧
  (x : ℤ) - (y : ℤ) + (z : ℤ) = 19 :=
sorry

end least_value_x_minus_y_plus_z_l3569_356982


namespace smallest_square_enclosing_circle_area_l3569_356901

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the area of the smallest enclosing square
def smallest_enclosing_square_area (r : ℝ) : ℝ := (2 * r) ^ 2

-- Theorem statement
theorem smallest_square_enclosing_circle_area :
  smallest_enclosing_square_area radius = 100 := by
  sorry

end smallest_square_enclosing_circle_area_l3569_356901


namespace burning_time_3x5_rectangle_l3569_356946

/-- Represents a rectangle of toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time to burn one toothpick

/-- Calculates the burning time for a ToothpickRectangle -/
def burningTime (rect : ToothpickRectangle) : Nat :=
  let maxDim := max rect.rows rect.cols
  (maxDim - 1) * rect.burnTime + 5

theorem burning_time_3x5_rectangle :
  let rect : ToothpickRectangle := {
    rows := 3,
    cols := 5,
    burnTime := 10
  }
  burningTime rect = 65 := by sorry

end burning_time_3x5_rectangle_l3569_356946


namespace quadratic_inequality_region_l3569_356919

theorem quadratic_inequality_region (x y : ℝ) :
  (∀ t : ℝ, t^2 ≤ 1 → t^2 + y*t + x ≥ 0) →
  (y ≤ x + 1 ∧ y ≥ -x - 1 ∧ x ≥ y^2/4) :=
by sorry

end quadratic_inequality_region_l3569_356919


namespace sum_of_digits_l3569_356915

/-- Given two single-digit numbers x and y, prove that x + y = 6 under certain conditions. -/
theorem sum_of_digits (x y : ℕ) : 
  (0 ≤ x ∧ x ≤ 9) →
  (0 ≤ y ∧ y ≤ 9) →
  (200 + 10 * x + 3) + 326 = (500 + 10 * y + 9) →
  (500 + 10 * y + 9) % 9 = 0 →
  x + y = 6 := by
sorry

end sum_of_digits_l3569_356915


namespace complex_equation_solution_l3569_356908

theorem complex_equation_solution (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 1))
  (eq2 : b = (a + c) / (y - 1))
  (eq3 : c = (a + b) / (z - 1))
  (eq4 : x * y + x * z + y * z = 7)
  (eq5 : x + y + z = 3) :
  x * y * z = 9 := by
  sorry


end complex_equation_solution_l3569_356908


namespace lukes_coin_piles_l3569_356988

theorem lukes_coin_piles (num_quarter_piles : ℕ) : 
  (∃ (num_dime_piles : ℕ), 
    num_quarter_piles = num_dime_piles ∧ 
    3 * num_quarter_piles + 3 * num_dime_piles = 30) → 
  num_quarter_piles = 5 := by
sorry

end lukes_coin_piles_l3569_356988


namespace bottle_caps_given_l3569_356989

theorem bottle_caps_given (initial_caps : Real) (remaining_caps : Real) 
  (h1 : initial_caps = 7.0)
  (h2 : remaining_caps = 5.0) :
  initial_caps - remaining_caps = 2.0 := by
sorry

end bottle_caps_given_l3569_356989


namespace parkway_elementary_soccer_l3569_356990

/-- The number of students playing soccer in the fifth grade at Parkway Elementary School -/
def students_playing_soccer (total_students : ℕ) (boys : ℕ) (boys_percentage : ℚ) (girls_not_playing : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of students playing soccer -/
theorem parkway_elementary_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (boys_percentage : ℚ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : boys_percentage = 86 / 100)
  (h4 : girls_not_playing = 89) :
  students_playing_soccer total_students boys boys_percentage girls_not_playing = 250 := by
    sorry

end parkway_elementary_soccer_l3569_356990


namespace age_relation_proof_l3569_356936

/-- Represents the current ages and future time when Alex's age is thrice Ben's --/
structure AgeRelation where
  ben_age : ℕ
  alex_age : ℕ
  michael_age : ℕ
  future_years : ℕ

/-- The conditions of the problem --/
def age_conditions (ar : AgeRelation) : Prop :=
  ar.ben_age = 4 ∧
  ar.alex_age = ar.ben_age + 30 ∧
  ar.michael_age = ar.alex_age + 4 ∧
  ar.alex_age + ar.future_years = 3 * (ar.ben_age + ar.future_years)

/-- The theorem to prove --/
theorem age_relation_proof :
  ∃ (ar : AgeRelation), age_conditions ar ∧ ar.future_years = 11 :=
sorry

end age_relation_proof_l3569_356936


namespace ice_skating_falls_l3569_356949

theorem ice_skating_falls (steven_falls sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : sonya_falls = 6) : 
  (steven_falls + 13) / 2 - sonya_falls = 2 := by
  sorry

end ice_skating_falls_l3569_356949


namespace xy_value_l3569_356964

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
sorry

end xy_value_l3569_356964


namespace square_difference_identity_nine_point_five_squared_l3569_356902

theorem square_difference_identity (x : ℝ) : (10 - x)^2 = 10^2 - 2 * 10 * x + x^2 := by sorry

theorem nine_point_five_squared :
  (9.5 : ℝ)^2 = 10^2 - 2 * 10 * 0.5 + 0.5^2 := by sorry

end square_difference_identity_nine_point_five_squared_l3569_356902


namespace expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l3569_356909

-- Define variables
variable (x y : ℝ)

-- First expression
def expr1 (x y : ℝ) : ℝ := 3*x^2*y - (2*x*y^2 - 2*(x*y - 1.5*x^2*y) + x*y) + 3*x*y^2

-- Second expression
def expr2 (x y : ℝ) : ℝ := (2*x + 3*y) - 4*y - (3*x - 2*y)

-- Theorem for the first expression
theorem expr1_simplification : 
  expr1 x y = x*y^2 + x*y := by sorry

-- Theorem for the evaluation of the first expression
theorem expr1_evaluation : 
  expr1 (-3) (-2) = -6 := by sorry

-- Theorem for the second expression
theorem expr2_simplification :
  expr2 x y = -x + y := by sorry

-- Theorem for the evaluation of the second expression
theorem expr2_evaluation :
  expr2 (-3) 2 = 5 := by sorry

end expr1_simplification_expr1_evaluation_expr2_simplification_expr2_evaluation_l3569_356909


namespace book_arrangement_count_book_arrangement_proof_l3569_356977

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 4 + 3 + 2
  let geometry_books : ℕ := 4
  let number_theory_books : ℕ := 3
  let algebra_books : ℕ := 2
  Nat.choose total_books geometry_books * 
  Nat.choose (total_books - geometry_books) number_theory_books * 
  Nat.choose (total_books - geometry_books - number_theory_books) algebra_books

theorem book_arrangement_proof : book_arrangement_count = 1260 := by
  sorry

end book_arrangement_count_book_arrangement_proof_l3569_356977


namespace product_difference_squares_divisible_by_three_l3569_356954

theorem product_difference_squares_divisible_by_three (m n : ℤ) :
  ∃ k : ℤ, m * n * (m^2 - n^2) = 3 * k := by
sorry

end product_difference_squares_divisible_by_three_l3569_356954


namespace expression_evaluation_l3569_356917

theorem expression_evaluation (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = -1/2) : 
  4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y) = 5/4 := by
  sorry

end expression_evaluation_l3569_356917


namespace gathering_handshakes_l3569_356966

/-- Calculates the number of handshakes in a gathering with specific rules -/
def handshakes (n : ℕ) : ℕ :=
  let couples := n
  let men := couples
  let women := couples
  let guest := 1
  let total_people := men + women + guest
  let handshakes_among_men := men * (men - 1) / 2
  let handshakes_men_women := men * (women - 1)
  let handshakes_with_guest := total_people - 1
  handshakes_among_men + handshakes_men_women + handshakes_with_guest

/-- Theorem stating that in a gathering of 15 married couples and 1 special guest,
    with specific handshake rules, the total number of handshakes is 345 -/
theorem gathering_handshakes : handshakes 15 = 345 := by
  sorry

end gathering_handshakes_l3569_356966


namespace rectangular_field_perimeter_l3569_356959

/-- The perimeter of a rectangular field with length 7/5 of its width and width of 80 meters is 384 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (length width : ℝ),
  length = (7/5) * width →
  width = 80 →
  2 * (length + width) = 384 :=
by
  sorry

end rectangular_field_perimeter_l3569_356959


namespace number_calculation_l3569_356900

theorem number_calculation (x : ℝ) : 0.2 * x = 0.4 * 140 + 80 → x = 680 := by
  sorry

end number_calculation_l3569_356900
