import Mathlib

namespace binomial_coefficient_ratio_l2908_290898

theorem binomial_coefficient_ratio (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 3 →
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end binomial_coefficient_ratio_l2908_290898


namespace book_pages_calculation_l2908_290836

theorem book_pages_calculation (total_pages : ℕ) : 
  (7 : ℚ) / 13 * total_pages + 
  (5 : ℚ) / 9 * ((6 : ℚ) / 13 * total_pages) + 
  96 = total_pages → 
  total_pages = 468 := by
sorry

end book_pages_calculation_l2908_290836


namespace valid_solutions_l2908_290895

def is_valid_solution (xyz : ℕ) : Prop :=
  xyz ≥ 100 ∧ xyz ≤ 999 ∧ (456000 + xyz) % 504 = 0

theorem valid_solutions :
  ∀ xyz : ℕ, is_valid_solution xyz ↔ (xyz = 120 ∨ xyz = 624) :=
sorry

end valid_solutions_l2908_290895


namespace not_always_zero_l2908_290896

-- Define the heart operation
def heart (x y : ℝ) : ℝ := |x + y|

-- Theorem stating that the statement is false
theorem not_always_zero : ¬ ∀ x : ℝ, heart x x = 0 := by
  sorry

end not_always_zero_l2908_290896


namespace cylinder_increase_equality_l2908_290829

theorem cylinder_increase_equality (x : ℝ) : 
  x > 0 → 
  π * (8 + x)^2 * 3 = π * 8^2 * (3 + x) → 
  x = 16/3 := by
sorry

end cylinder_increase_equality_l2908_290829


namespace su_buqing_star_distance_l2908_290803

theorem su_buqing_star_distance (distance : ℝ) : 
  distance = 218000000 → distance = 2.18 * (10 ^ 8) := by
  sorry

end su_buqing_star_distance_l2908_290803


namespace target_average_income_l2908_290870

def past_incomes : List ℝ := [406, 413, 420, 436, 395]
def next_two_weeks_avg : ℝ := 365
def total_weeks : ℕ := 7

theorem target_average_income :
  let total_past_income := past_incomes.sum
  let total_next_two_weeks := 2 * next_two_weeks_avg
  let total_income := total_past_income + total_next_two_weeks
  total_income / total_weeks = 400 := by
  sorry

end target_average_income_l2908_290870


namespace min_value_of_reciprocal_sum_l2908_290842

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop := a * x - b * y + 8 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem min_value_of_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_line_passes_center : line_eq a b (circle_center.1) (circle_center.2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → line_eq a' b' (circle_center.1) (circle_center.2) → 
    1/a + 1/b ≤ 1/a' + 1/b') ∧ 1/a + 1/b = 1 :=
by sorry

end min_value_of_reciprocal_sum_l2908_290842


namespace women_who_left_l2908_290830

theorem women_who_left (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  ∃ (left : ℕ), 2 * (initial_women - left) = 24 ∧ left = 3 :=
by sorry

end women_who_left_l2908_290830


namespace certain_number_proof_l2908_290850

theorem certain_number_proof (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 200 → N = 384 := by
  sorry

end certain_number_proof_l2908_290850


namespace one_minus_repeating_8_l2908_290801

/-- The value of the repeating decimal 0.888... -/
def repeating_decimal_8 : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_8 : 1 - repeating_decimal_8 = 1/9 := by
  sorry

end one_minus_repeating_8_l2908_290801


namespace sum_of_squares_l2908_290844

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2*n*b^2 = k^2) :
  ∃ x y : ℕ, a^2 + n*b^2 = x^2 + y^2 := by
sorry

end sum_of_squares_l2908_290844


namespace max_value_of_f_l2908_290884

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)

noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem max_value_of_f (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (-π/6) (π/3), f x₀ m = -4) →
  (∃ x₁ ∈ Set.Icc (-π/6) (π/3), ∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≤ f x₁ m) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≤ -3/2) ∧
  f (π/6) m = -3/2 :=
by sorry

end max_value_of_f_l2908_290884


namespace least_possible_difference_l2908_290828

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → 
  Odd z → 
  (∀ d : ℤ, d = z - x → d ≥ 9) ∧ (∃ x' y' z' : ℤ, x' < y' ∧ y' < z' ∧ y' - x' > 5 ∧ Even x' ∧ Odd y' ∧ Odd z' ∧ z' - x' = 9) :=
by sorry

end least_possible_difference_l2908_290828


namespace equation_solution_l2908_290814

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end equation_solution_l2908_290814


namespace lava_lamp_probability_l2908_290848

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_turned_on : ℕ := 4

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps
  let color_condition := Nat.choose (total_lamps - 4) (red_lamps - 2)
  let on_off_condition := Nat.choose (total_lamps - 2) (lamps_turned_on - 2)
  (color_condition * on_off_condition : ℚ) / (total_arrangements * total_arrangements) = 225 / 4900 := by
  sorry

end lava_lamp_probability_l2908_290848


namespace ship_grain_calculation_l2908_290874

/-- The amount of grain spilled into the water, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining onboard, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 :=
sorry

end ship_grain_calculation_l2908_290874


namespace distance_focus_to_asymptote_l2908_290815

/-- The distance from the right focus of the hyperbola x²/4 - y² = 1 to its asymptote x - 2y = 0 is 1 -/
theorem distance_focus_to_asymptote (x y : ℝ) : 
  let hyperbola := (x^2 / 4 - y^2 = 1)
  let right_focus := (x = Real.sqrt 5 ∧ y = 0)
  let asymptote := (x - 2*y = 0)
  let distance := |x - 2*y| / Real.sqrt 5
  (hyperbola ∧ right_focus ∧ asymptote) → distance = 1 := by
sorry


end distance_focus_to_asymptote_l2908_290815


namespace leaf_movement_l2908_290847

theorem leaf_movement (forward_distance : ℕ) (num_gusts : ℕ) (total_distance : ℕ) 
  (h1 : forward_distance = 5)
  (h2 : num_gusts = 11)
  (h3 : total_distance = 33) :
  ∃ (backward_distance : ℕ), 
    num_gusts * (forward_distance - backward_distance) = total_distance ∧ 
    backward_distance = 2 :=
by sorry

end leaf_movement_l2908_290847


namespace only_B_is_random_l2908_290831

-- Define the type for events
inductive Event
| A  -- A coin thrown from the ground will fall down
| B  -- A shooter hits the target with 10 points in one shot
| C  -- The sun rises from the east
| D  -- A horse runs at a speed of 70 meters per second

-- Define what it means for an event to be random
def is_random (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem stating that only event B is random
theorem only_B_is_random :
  ∀ e : Event, is_random e ↔ e = Event.B :=
by
  sorry


end only_B_is_random_l2908_290831


namespace correct_propositions_l2908_290835

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations and operations
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p₁ p₂ : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p₁ p₂ : Plane) : Prop := sorry
def intersection (p₁ p₂ : Plane) : Line := sorry

-- State the theorem
theorem correct_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ (m n : Line) (α β : Plane) (l : Line),
    subset m α → subset n β → perpendicular_planes α β → 
    intersection α β = l → perpendicular m l → perpendicular m n) ∧
  (∀ (m : Line) (α β : Plane),
    perpendicular m α → perpendicular m β → parallel_planes α β) := by
  sorry

end correct_propositions_l2908_290835


namespace circle_center_l2908_290871

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 9

/-- Theorem: The center of the circle is (1, 3) -/
theorem circle_center : is_center 1 3 := by
  sorry

end circle_center_l2908_290871


namespace gumball_difference_l2908_290840

theorem gumball_difference (x : ℤ) : 
  (19 * 3 ≤ 16 + 12 + x ∧ 16 + 12 + x ≤ 25 * 3) →
  (∃ (max min : ℤ), 
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → y ≤ max) ∧
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → min ≤ y) ∧
    max - min = 18) :=
by sorry

end gumball_difference_l2908_290840


namespace sock_matching_probability_l2908_290821

def total_socks : ℕ := 18
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_combinations : ℕ := total_socks.choose 2
def matching_gray_combinations : ℕ := gray_socks.choose 2
def matching_white_combinations : ℕ := white_socks.choose 2
def matching_combinations : ℕ := matching_gray_combinations + matching_white_combinations

theorem sock_matching_probability :
  (matching_combinations : ℚ) / total_combinations = 73 / 153 := by sorry

end sock_matching_probability_l2908_290821


namespace election_winner_votes_l2908_290841

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 288) :
  (62 : ℚ) / 100 * total_votes = 744 := by
sorry

end election_winner_votes_l2908_290841


namespace intersection_point_is_solution_l2908_290859

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (78/19, 41/19)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*x - 2*y = 8

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 3*y = 27

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end intersection_point_is_solution_l2908_290859


namespace equal_probabilities_after_adding_balls_l2908_290888

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  yellow : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.yellow)

/-- The initial contents of the bag -/
def initialBag : BagContents := ⟨2, 3⟩

/-- The contents of the bag after adding balls -/
def finalBag : BagContents := ⟨initialBag.white + 4, initialBag.yellow + 3⟩

/-- Theorem stating that the probabilities are equal after adding balls -/
theorem equal_probabilities_after_adding_balls :
  probability finalBag finalBag.white = probability finalBag finalBag.yellow := by
  sorry

end equal_probabilities_after_adding_balls_l2908_290888


namespace child_tickets_sold_l2908_290812

/-- Proves the number of child tickets sold given ticket prices and total sales information -/
theorem child_tickets_sold 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sales : ℕ) 
  (total_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : total_tickets = 42) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_sales ∧
    child_tickets = 16 :=
by sorry

end child_tickets_sold_l2908_290812


namespace circle_equation_m_range_l2908_290865

theorem circle_equation_m_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m - 3)*x + 2*y + 5 = 0 → ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m > 5 ∨ m < 1) :=
sorry

end circle_equation_m_range_l2908_290865


namespace problem_1_l2908_290825

theorem problem_1 : (-3) + (-9) - 10 - (-18) = -4 := by
  sorry

end problem_1_l2908_290825


namespace seventh_term_is_2187_l2908_290839

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ → ℕ  -- The sequence
  r : ℕ      -- The common ratio
  first_term : a 1 = 3
  ratio_def : ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_is_2187 (seq : GeometricSequence) (h : seq.a 6 = 972) :
  seq.a 7 = 2187 := by
  sorry

end seventh_term_is_2187_l2908_290839


namespace rectangle_area_difference_main_theorem_l2908_290869

theorem rectangle_area_difference : ℕ → Prop :=
fun diff =>
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 225) ∧  -- Largest area
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 29) ∧  -- Smallest area
  diff = 225 - 29

theorem main_theorem : rectangle_area_difference 196 := by
  sorry

end rectangle_area_difference_main_theorem_l2908_290869


namespace inequality_solution_set_l2908_290807

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - a) * (x - 2*a) < 0}
  S = if a < 0 then Set.Ioo (2*a) a
      else if a = 0 then ∅
      else Set.Ioo a (2*a) := by sorry

end inequality_solution_set_l2908_290807


namespace characterization_of_solutions_l2908_290881

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) * (f x - f y) = (x - y) * f x * f y

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_solutions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ C : ℝ, C ≠ 0 ∧ ∀ x : ℝ, f x = C * x :=
by sorry

end characterization_of_solutions_l2908_290881


namespace keystone_arch_angle_l2908_290808

theorem keystone_arch_angle (n : ℕ) (angle : ℝ) : 
  n = 10 → -- There are 10 trapezoids
  angle = (180 : ℝ) - (360 / (2 * n)) → -- The larger interior angle
  angle = 99 := by
  sorry

end keystone_arch_angle_l2908_290808


namespace same_day_after_313_weeks_l2908_290813

/-- The day of the week is represented as an integer from 0 to 6 -/
def DayOfWeek := Fin 7

/-- The number of weeks that have passed -/
def weeks : ℕ := 313

/-- Given an initial day of the week, returns the day of the week after a specified number of weeks -/
def day_after_weeks (initial_day : DayOfWeek) (n : ℕ) : DayOfWeek :=
  ⟨(initial_day.val + 7 * n) % 7, by sorry⟩

/-- Theorem: The day of the week remains the same after exactly 313 weeks -/
theorem same_day_after_313_weeks (d : DayOfWeek) : 
  day_after_weeks d weeks = d := by sorry

end same_day_after_313_weeks_l2908_290813


namespace range_of_f_l2908_290868

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6*x
  else 0  -- We define f as 0 outside the given intervals

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-9 : ℝ) 1 := by sorry

end range_of_f_l2908_290868


namespace equation_solution_l2908_290809

theorem equation_solution : ∃ x : ℚ, (5/100 * x + 12/100 * (30 + x) = 144/10) ∧ x = 108/17 := by
  sorry

end equation_solution_l2908_290809


namespace equation_graph_is_axes_l2908_290862

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = T := by sorry

end equation_graph_is_axes_l2908_290862


namespace ice_cream_sales_theorem_l2908_290852

/-- Calculates the total number of ice cream cones sold in a week based on given sales pattern -/
def total_ice_cream_sales (monday : ℕ) (tuesday : ℕ) : ℕ :=
  let wednesday := 2 * tuesday
  let thursday := (3 * wednesday) / 2
  let friday := (3 * thursday) / 4
  let weekend := 2 * friday
  monday + tuesday + wednesday + thursday + friday + weekend

/-- Theorem stating that the total ice cream sales for the week is 163,000 -/
theorem ice_cream_sales_theorem : total_ice_cream_sales 10000 12000 = 163000 := by
  sorry

#eval total_ice_cream_sales 10000 12000

end ice_cream_sales_theorem_l2908_290852


namespace industrial_machine_output_l2908_290805

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  totalShirts : ℕ
  workingMinutes : ℕ

/-- Calculate the shirts per minute for a given machine -/
def shirtsPerMinute (machine : ShirtMachine) : ℚ :=
  machine.totalShirts / machine.workingMinutes

theorem industrial_machine_output (machine : ShirtMachine) 
  (h1 : machine.totalShirts = 6)
  (h2 : machine.workingMinutes = 2) : 
  shirtsPerMinute machine = 3 := by
  sorry

end industrial_machine_output_l2908_290805


namespace hyperbola_equation_l2908_290818

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ x = 1 ∧ y = 0) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 4 = 1) :=
by sorry

end hyperbola_equation_l2908_290818


namespace B_power_99_l2908_290816

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 0, 0;
     0, 0, 1;
     0, -1, 0]

theorem B_power_99 : B^99 = B := by sorry

end B_power_99_l2908_290816


namespace multiple_calculation_l2908_290883

theorem multiple_calculation (number : ℝ) (value : ℝ) (multiple : ℝ) : 
  number = -4.5 →
  value = 36 →
  10 * number = value - multiple * number →
  multiple = -18 := by
sorry

end multiple_calculation_l2908_290883


namespace perpendicular_line_equation_l2908_290837

/-- Given a line L1 with equation 4x - 2y + 1 = 0, prove that the line L2 passing through
    the point (2, -3) and perpendicular to L1 has the equation x + 2y + 4 = 0. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x - 2 * y + 1 = 0
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y + 4 = 0
  (∀ x y, L1 x y → L2 x y → (y - P.2) = -(x - P.1)) →
  (∀ x y, L1 x y → L2 x y → (y - P.2) * (x - P.1) = -1) →
  L2 P.1 P.2 :=
by
  sorry


end perpendicular_line_equation_l2908_290837


namespace abrahams_shopping_budget_l2908_290864

/-- Abraham's shopping problem -/
theorem abrahams_shopping_budget (budget : ℕ) 
  (shower_gel_price shower_gel_quantity : ℕ) 
  (toothpaste_price laundry_detergent_price : ℕ) : 
  budget = 60 →
  shower_gel_price = 4 →
  shower_gel_quantity = 4 →
  toothpaste_price = 3 →
  laundry_detergent_price = 11 →
  budget - (shower_gel_price * shower_gel_quantity + toothpaste_price + laundry_detergent_price) = 30 := by
  sorry


end abrahams_shopping_budget_l2908_290864


namespace negation_of_existence_l2908_290873

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end negation_of_existence_l2908_290873


namespace sum_of_coefficients_l2908_290851

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^6 + 9 * x^3 - 6) + 8 * (x^4 - 6 * x^2 + 3)

theorem sum_of_coefficients : (polynomial 1) = 4 := by
  sorry

end sum_of_coefficients_l2908_290851


namespace travel_options_count_l2908_290800

/-- The number of travel options from A to C given the number of trains from A to B and ferries from B to C -/
def travelOptions (trains : ℕ) (ferries : ℕ) : ℕ :=
  trains * ferries

/-- Theorem stating that the number of travel options from A to C is 6 -/
theorem travel_options_count :
  travelOptions 3 2 = 6 := by
  sorry

end travel_options_count_l2908_290800


namespace fixed_point_of_line_l2908_290811

theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end fixed_point_of_line_l2908_290811


namespace x_plus_2y_equals_10_l2908_290866

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 3*y = 1) : 
  x + 2*y = 10 := by
sorry

end x_plus_2y_equals_10_l2908_290866


namespace base7_5213_to_base10_l2908_290885

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- The base 10 representation of 5213₇ is 1823 -/
theorem base7_5213_to_base10 : base7ToBase10 5 2 1 3 = 1823 := by
  sorry

end base7_5213_to_base10_l2908_290885


namespace lunch_total_amount_l2908_290806

/-- The total amount spent on lunch given the conditions -/
theorem lunch_total_amount (your_spending friend_spending : ℕ) 
  (h1 : friend_spending = 10)
  (h2 : friend_spending = your_spending + 3) : 
  your_spending + friend_spending = 17 := by
  sorry

end lunch_total_amount_l2908_290806


namespace appropriate_word_count_appropriate_lengths_l2908_290878

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 30 }

/-- The optimal speaking rate in words per minute -/
def OptimalSpeakingRate : ℝ := 135

/-- Calculates the number of words for a given duration at the optimal speaking rate -/
def WordCount (duration : PresentationDuration) : ℝ :=
  duration.val * OptimalSpeakingRate

/-- Theorem stating that the appropriate word count is between 2700 and 4050 -/
theorem appropriate_word_count (duration : PresentationDuration) :
  2700 ≤ WordCount duration ∧ WordCount duration ≤ 4050 := by
  sorry

/-- Theorem stating that 3000 and 3700 words are appropriate lengths for the presentation -/
theorem appropriate_lengths :
  ∃ (d1 d2 : PresentationDuration), WordCount d1 = 3000 ∧ WordCount d2 = 3700 := by
  sorry

end appropriate_word_count_appropriate_lengths_l2908_290878


namespace pastry_combinations_l2908_290832

/-- The number of ways to select pastries -/
def select_pastries (total : ℕ) (types : ℕ) : ℕ :=
  if types > total then 0
  else
    let remaining := total - types
    -- Ways to distribute remaining pastries among types
    (types^remaining + types * (types - 1) * remaining + Nat.choose types remaining) / Nat.factorial remaining

/-- Theorem: Selecting 8 pastries from 5 types, with at least one of each type, results in 25 combinations -/
theorem pastry_combinations : select_pastries 8 5 = 25 := by
  sorry


end pastry_combinations_l2908_290832


namespace pie_cost_is_six_l2908_290810

/-- The cost of a pie given initial and remaining amounts -/
def pieCost (initialAmount remainingAmount : ℕ) : ℕ :=
  initialAmount - remainingAmount

/-- Theorem: The cost of the pie is $6 -/
theorem pie_cost_is_six :
  pieCost 63 57 = 6 := by
  sorry

end pie_cost_is_six_l2908_290810


namespace murtha_pebble_collection_l2908_290875

/-- Calculates the sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the sum of an arithmetic sequence --/
def sum_arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles --/
def total_days : ℕ := 15

/-- The number of days Murtha skips collecting pebbles --/
def skipped_days : ℕ := total_days / 3

/-- Theorem: Murtha's pebble collection after 15 days --/
theorem murtha_pebble_collection :
  sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days = 75 := by
  sorry

#eval sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days

end murtha_pebble_collection_l2908_290875


namespace number_puzzle_l2908_290863

theorem number_puzzle (x : ℝ) : (((3/4 * x) - 25) / 7) + 50 = 100 → x = 500 := by
  sorry

end number_puzzle_l2908_290863


namespace cone_sphere_ratio_l2908_290819

/-- Given a sphere and a right circular cone, if the volume of the cone is one-third
    that of the sphere, and the radius of the base of the cone is twice the radius of the sphere,
    then the ratio of the altitude of the cone to the radius of its base is 1/6. -/
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (h_pos : 0 < r) : 
  (4 / 3 * Real.pi * r^3) / 3 = 1 / 3 * Real.pi * (2 * r)^2 * h →
  h / (2 * r) = 1 / 6 := by
  sorry

#check cone_sphere_ratio

end cone_sphere_ratio_l2908_290819


namespace line_equations_l2908_290872

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Define the line passing through the origin
def line_through_origin (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line parallel to 2x + y + 5 = 0
def parallel_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line perpendicular to 2x + y + 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 5 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (line_through_origin M.1 M.2) ∧
  (parallel_line M.1 M.2) ∧
  (perpendicular_line M.1 M.2) := by sorry

end line_equations_l2908_290872


namespace mike_bought_two_for_friend_l2908_290843

/-- Represents the problem of calculating the number of rose bushes Mike bought for his friend. -/
def mike_rose_bushes_for_friend 
  (total_rose_bushes : ℕ)
  (rose_bush_price : ℕ)
  (total_aloes : ℕ)
  (aloe_price : ℕ)
  (spent_on_self : ℕ) : ℕ :=
  total_rose_bushes - (spent_on_self - total_aloes * aloe_price) / rose_bush_price

/-- Theorem stating that Mike bought 2 rose bushes for his friend. -/
theorem mike_bought_two_for_friend :
  mike_rose_bushes_for_friend 6 75 2 100 500 = 2 := by
  sorry

end mike_bought_two_for_friend_l2908_290843


namespace range_of_a_l2908_290820

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2 * x * (x - a) < 1) → a > -1 := by
  sorry

end range_of_a_l2908_290820


namespace statistics_test_probability_l2908_290867

def word : String := "STATISTICS"
def test_word : String := "TEST"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem statistics_test_probability :
  let total_tiles := word.length
  let overlapping_tiles := (test_word.toList.eraseDups.filter (λ c => word.contains c))
                            |>.map (λ c => letter_count word c)
                            |>.sum
  (↑overlapping_tiles : ℚ) / total_tiles = 1 / 2 := by
  sorry

end statistics_test_probability_l2908_290867


namespace vector_sum_coordinates_l2908_290889

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coordinates : 2 • a + b = (-3, 4) := by sorry

end vector_sum_coordinates_l2908_290889


namespace octal_67_equals_ternary_2001_l2908_290824

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to ternary --/
def decimal_to_ternary (n : ℕ) : ℕ := sorry

theorem octal_67_equals_ternary_2001 :
  decimal_to_ternary (octal_to_decimal 67) = 2001 := by sorry

end octal_67_equals_ternary_2001_l2908_290824


namespace andrews_age_l2908_290833

/-- Proves that Andrew's current age is 30, given the donation information -/
theorem andrews_age (donation_start_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) :
  donation_start_age = 11 →
  annual_donation = 7 →
  total_donation = 133 →
  donation_start_age + (total_donation / annual_donation) = 30 :=
by sorry

end andrews_age_l2908_290833


namespace digit_equation_solution_l2908_290879

theorem digit_equation_solution : ∃ (Θ : ℕ), 
  Θ ≤ 9 ∧ 
  252 / Θ = 40 + 2 * Θ ∧ 
  Θ = 5 := by
  sorry

end digit_equation_solution_l2908_290879


namespace sin_shift_left_l2908_290893

theorem sin_shift_left (x : ℝ) : 
  Real.sin (x + π/4) = Real.sin (x - (-π/4)) := by sorry

end sin_shift_left_l2908_290893


namespace smallest_sum_arithmetic_geometric_l2908_290890

theorem smallest_sum_arithmetic_geometric (A B C D : ℤ) : 
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →           -- B, C, D form a geometric sequence
  (C = (4 * B) / 3) →         -- C/B = 4/3
  (∀ A' B' C' D' : ℤ, 
    (∃ d' : ℤ, C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (4 * B') / 3) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end smallest_sum_arithmetic_geometric_l2908_290890


namespace distance_not_unique_l2908_290857

/-- Given two segments AB and BC with lengths 4 and 3 respectively, 
    prove that the length of AC cannot be uniquely determined. -/
theorem distance_not_unique (A B C : ℝ × ℝ) 
  (hAB : dist A B = 4) 
  (hBC : dist B C = 3) : 
  ¬ ∃! d, dist A C = d :=
sorry

end distance_not_unique_l2908_290857


namespace residue_calculation_l2908_290887

theorem residue_calculation : (240 * 15 - 21 * 9 + 6) % 18 = 15 := by
  sorry

end residue_calculation_l2908_290887


namespace digit_sum_subtraction_l2908_290827

theorem digit_sum_subtraction (n : ℕ) : 
  2010 ≤ n ∧ n ≤ 2019 → n - (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2007 := by
  sorry

end digit_sum_subtraction_l2908_290827


namespace sloth_shoe_theorem_l2908_290845

/-- The number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- The number of complete sets of shoes desired -/
def desired_sets : ℕ := 5

/-- The number of sets of shoes already owned -/
def owned_sets : ℕ := 1

/-- Calculate the number of pairs of shoes needed to be purchased -/
def shoes_to_buy : ℕ :=
  (desired_sets * sloth_feet - owned_sets * sloth_feet) / 2

theorem sloth_shoe_theorem : shoes_to_buy = 6 := by
  sorry

end sloth_shoe_theorem_l2908_290845


namespace sqrt_29_between_consecutive_integers_product_l2908_290823

theorem sqrt_29_between_consecutive_integers_product (n m : ℤ) :
  n < m ∧ m = n + 1 ∧ (n : ℝ) < Real.sqrt 29 ∧ Real.sqrt 29 < (m : ℝ) →
  n * m = 30 :=
sorry

end sqrt_29_between_consecutive_integers_product_l2908_290823


namespace exists_points_D_E_l2908_290822

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is on a line segment
def isOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem exists_points_D_E (ABC : Triangle) : 
  ∃ (D E : ℝ × ℝ), 
    isOnSegment D ABC.A ABC.B ∧ 
    isOnSegment E ABC.A ABC.C ∧ 
    distance ABC.A D = distance D E ∧ 
    distance D E = distance E ABC.C := by
  sorry

end exists_points_D_E_l2908_290822


namespace ceiling_product_equation_l2908_290826

theorem ceiling_product_equation :
  ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end ceiling_product_equation_l2908_290826


namespace correct_calculation_l2908_290860

theorem correct_calculation (x : ℝ) : 8 * x + 8 = 56 → (x / 8) + 7 = 7.75 := by
  sorry

end correct_calculation_l2908_290860


namespace fixed_point_theorem_l2908_290861

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the property of a line passing through K
def line_through_K (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the intersection points A and B
def intersection_points (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_K m x₁ y₁ ∧ line_through_K m x₂ y₂ ∧
  y₁ ≠ y₂

-- Define point D as symmetric to A with respect to x-axis
def point_D (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- The main theorem
theorem fixed_point_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  m ≠ 0 →
  intersection_points m x₁ y₁ x₂ y₂ →
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧ 
    F.1 = (1 - t) * x₂ + t * (point_D x₁ y₁).1 ∧
    F.2 = (1 - t) * y₂ + t * (point_D x₁ y₁).2 :=
sorry

end fixed_point_theorem_l2908_290861


namespace circle_center_radius_sum_l2908_290897

/-- Given a circle C with equation x^2 - 8y - 7 = -y^2 - 6x, 
    prove that the sum of its center coordinates and radius is 1 + 4√2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*y - 7 = -y^2 - 6*x → (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = 1 + 4 * Real.sqrt 2 :=
by sorry

end circle_center_radius_sum_l2908_290897


namespace fractional_method_min_experiments_l2908_290849

/-- The number of division points in the temperature range -/
def division_points : ℕ := 33

/-- The minimum number of experiments needed -/
def min_experiments : ℕ := 7

/-- Theorem stating the minimum number of experiments needed for the given conditions -/
theorem fractional_method_min_experiments :
  ∃ (n : ℕ), 2^n - 1 ≥ division_points ∧ n = min_experiments :=
sorry

end fractional_method_min_experiments_l2908_290849


namespace expand_expression_l2908_290886

theorem expand_expression (x : ℝ) : (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 := by
  sorry

end expand_expression_l2908_290886


namespace exam_attendance_calculation_l2908_290858

theorem exam_attendance_calculation (total_topics : ℕ) 
  (all_topics_pass_percent : ℚ) (no_topic_pass_percent : ℚ)
  (one_topic_pass_percent : ℚ) (two_topics_pass_percent : ℚ)
  (four_topics_pass_percent : ℚ) (three_topics_pass_count : ℕ)
  (h1 : total_topics = 5)
  (h2 : all_topics_pass_percent = 1/10)
  (h3 : no_topic_pass_percent = 1/10)
  (h4 : one_topic_pass_percent = 1/5)
  (h5 : two_topics_pass_percent = 1/4)
  (h6 : four_topics_pass_percent = 6/25)
  (h7 : three_topics_pass_count = 500) :
  ∃ total_students : ℕ, total_students = 4546 ∧
  (all_topics_pass_percent + no_topic_pass_percent + one_topic_pass_percent + 
   two_topics_pass_percent + four_topics_pass_percent) * total_students + 
   three_topics_pass_count = total_students :=
by sorry

end exam_attendance_calculation_l2908_290858


namespace smallest_n_divisibility_l2908_290882

theorem smallest_n_divisibility (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 360 → (¬(54 ∣ m^2) ∨ ¬(1280 ∣ m^3))) ∧ 
  (54 ∣ 360^2) ∧ (1280 ∣ 360^3) := by
  sorry

end smallest_n_divisibility_l2908_290882


namespace kenny_mushroom_pieces_l2908_290853

/-- The number of mushroom pieces Kenny used on his pizza -/
def kenny_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) (karla_pieces : ℕ) (remaining_pieces : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (karla_pieces + remaining_pieces)

/-- Theorem stating the number of mushroom pieces Kenny used -/
theorem kenny_mushroom_pieces :
  kenny_pieces 22 4 42 8 = 38 := by
  sorry

end kenny_mushroom_pieces_l2908_290853


namespace reading_pages_in_week_l2908_290877

/-- Calculates the total number of pages read in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

/-- Theorem: Reading 5 pages in the morning and 10 pages in the evening for a week results in 105 pages read -/
theorem reading_pages_in_week :
  pages_read_in_week 5 10 7 = 105 := by
  sorry

end reading_pages_in_week_l2908_290877


namespace smaller_city_size_l2908_290894

/-- Proves that given a population density of 80 people per cubic yard, 
    if a larger city with 9000 cubic yards has 208000 more people than a smaller city, 
    then the smaller city has 6400 cubic yards. -/
theorem smaller_city_size (density : ℕ) (larger_city_size : ℕ) (population_difference : ℕ) :
  density = 80 →
  larger_city_size = 9000 →
  population_difference = 208000 →
  (larger_city_size * density) - (population_difference) = 6400 * density :=
by
  sorry

end smaller_city_size_l2908_290894


namespace largest_sample_number_l2908_290802

def systematic_sampling (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  let sample_size := total / interval
  start + interval * (sample_size - 1)

theorem largest_sample_number :
  systematic_sampling 500 7 25 = 482 := by
  sorry

end largest_sample_number_l2908_290802


namespace sqrt_expressions_equality_l2908_290880

theorem sqrt_expressions_equality : 
  (Real.sqrt 8 - Real.sqrt (1/2) + Real.sqrt 18 = (9 * Real.sqrt 2) / 2) ∧ 
  ((Real.sqrt 2 + Real.sqrt 3)^2 - Real.sqrt 24 = 5) := by
  sorry

end sqrt_expressions_equality_l2908_290880


namespace comic_book_collections_l2908_290804

/-- Kymbrea's initial comic book collection -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book collection rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book collection rate -/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection is twice Kymbrea's -/
def months : ℕ := 80

theorem comic_book_collections : 
  lashawn_initial + lashawn_rate * months = 2 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end comic_book_collections_l2908_290804


namespace barbara_shopping_l2908_290899

/-- The amount spent on goods other than tuna and water in Barbara's shopping trip -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_cost : ℚ) : ℚ :=
  total_cost - (tuna_packs * tuna_price + water_bottles * water_price)

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end barbara_shopping_l2908_290899


namespace combined_selling_price_l2908_290891

/-- Calculate the combined selling price of two articles given their costs, desired profits, tax rate, and packaging fees. -/
theorem combined_selling_price
  (cost_A cost_B : ℚ)
  (profit_rate_A profit_rate_B : ℚ)
  (tax_rate : ℚ)
  (packaging_fee : ℚ) :
  cost_A = 500 →
  cost_B = 800 →
  profit_rate_A = 1/10 →
  profit_rate_B = 3/20 →
  tax_rate = 1/20 →
  packaging_fee = 50 →
  ∃ (selling_price : ℚ),
    selling_price = 
      (cost_A + cost_A * profit_rate_A) * (1 + tax_rate) + packaging_fee +
      (cost_B + cost_B * profit_rate_B) * (1 + tax_rate) + packaging_fee ∧
    selling_price = 1643.5 := by
  sorry

#check combined_selling_price

end combined_selling_price_l2908_290891


namespace room_length_l2908_290846

/-- Given a rectangular room with width 4 meters and a paving cost of 950 per square meter
    resulting in a total cost of 20900, the length of the room is 5.5 meters. -/
theorem room_length (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 950 →
  total_cost = 20900 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
sorry


end room_length_l2908_290846


namespace chocolate_pieces_per_box_l2908_290834

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 14)
  (h2 : given_boxes = 8)
  (h3 : remaining_pieces = 18)
  (h4 : total_boxes > given_boxes) :
  (remaining_pieces / (total_boxes - given_boxes) : ℕ) = 3 := by
sorry

end chocolate_pieces_per_box_l2908_290834


namespace min_reciprocal_sum_l2908_290838

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end min_reciprocal_sum_l2908_290838


namespace box_width_l2908_290876

/-- The width of a rectangular box given its dimensions and cube properties -/
theorem box_width (length height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h1 : length = 10)
  (h2 : height = 4)
  (h3 : cube_volume = 12)
  (h4 : min_cubes = 60) :
  (min_cubes : ℝ) * cube_volume / (length * height) = 18 := by
  sorry

end box_width_l2908_290876


namespace bucket_capacity_reduction_l2908_290856

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end bucket_capacity_reduction_l2908_290856


namespace cost_of_one_milk_carton_l2908_290855

/-- The cost of 1 one-litre carton of milk, given that 4 cartons cost $4.88 -/
theorem cost_of_one_milk_carton :
  let total_cost : ℚ := 488/100  -- $4.88 represented as a rational number
  let num_cartons : ℕ := 4
  let cost_per_carton : ℚ := total_cost / num_cartons
  cost_per_carton = 122/100  -- $1.22 represented as a rational number
:= by sorry

end cost_of_one_milk_carton_l2908_290855


namespace sum_of_cubes_values_l2908_290817

open Complex Matrix

/-- A 3x3 circulant matrix with complex entries a, b, c -/
def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  !![a, b, c; b, c, a; c, a, b]

/-- The theorem statement -/
theorem sum_of_cubes_values (a b c : ℂ) : 
  M a b c ^ 2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry


end sum_of_cubes_values_l2908_290817


namespace dannys_chickens_l2908_290892

/-- Calculates the number of chickens on Dany's farm -/
theorem dannys_chickens (cows sheep : ℕ) (cow_sheep_bushels chicken_bushels total_bushels : ℕ) : 
  cows = 4 →
  sheep = 3 →
  cow_sheep_bushels = 2 →
  chicken_bushels = 3 →
  total_bushels = 35 →
  (cows + sheep) * cow_sheep_bushels + (total_bushels - (cows + sheep) * cow_sheep_bushels) / chicken_bushels = 7 := by
  sorry

end dannys_chickens_l2908_290892


namespace bobs_grade_is_35_l2908_290854

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℚ := jason_grade / 2

theorem bobs_grade_is_35 : bob_grade = 35 := by sorry

end bobs_grade_is_35_l2908_290854
