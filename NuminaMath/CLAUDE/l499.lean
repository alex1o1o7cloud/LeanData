import Mathlib

namespace tiles_required_for_room_floor_l499_49951

def room_length : Real := 6.24
def room_width : Real := 4.32
def tile_side : Real := 0.30

theorem tiles_required_for_room_floor :
  ⌈(room_length * room_width) / (tile_side * tile_side)⌉ = 300 := by
  sorry

end tiles_required_for_room_floor_l499_49951


namespace circle_equation_l499_49955

-- Define the point P
def P : ℝ × ℝ := (3, 1)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the two possible circle equations
def circle₁ (x y : ℝ) : Prop := (x - 4/5)^2 + (y - 3/5)^2 = 5
def circle₂ (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ → Prop), c P ∧
    (∀ (x y : ℝ), l₁ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε)))) ∧
    (∀ (x y : ℝ), l₂ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε))))) →
  circle₁ x y ∨ circle₂ x y :=
sorry

end circle_equation_l499_49955


namespace purely_imaginary_condition_l499_49922

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 → 
  (Complex.re ((1 - a * Complex.I) * (3 + 2 * Complex.I)) = 0 ∧
   Complex.im ((1 - a * Complex.I) * (3 + 2 * Complex.I)) ≠ 0) → 
  a = -3/2 := by
sorry

end purely_imaginary_condition_l499_49922


namespace hyperbola_foci_l499_49958

theorem hyperbola_foci (x y : ℝ) :
  (x^2 / 3 - y^2 / 4 = 1) →
  (∃ f : ℝ, f = Real.sqrt 7 ∧ 
    ((x = f ∧ y = 0) ∨ (x = -f ∧ y = 0)) →
    (x^2 / 3 - y^2 / 4 = 1)) :=
by sorry

end hyperbola_foci_l499_49958


namespace newspapers_sold_l499_49928

theorem newspapers_sold (magazines : ℕ) (total : ℕ) (newspapers : ℕ) : 
  magazines = 425 → total = 700 → newspapers = total - magazines → newspapers = 275 := by
  sorry

end newspapers_sold_l499_49928


namespace trig_simplification_l499_49967

theorem trig_simplification (α : ℝ) : 
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α := by
  sorry

end trig_simplification_l499_49967


namespace line_plane_perpendicular_l499_49907

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular (l m : Line) (α : Plane) :
  perpendicular l α → notParallel l m → perpendicular m α := by
  sorry

end line_plane_perpendicular_l499_49907


namespace expand_polynomial_l499_49950

theorem expand_polynomial (x : ℝ) : (7 * x^3 - 5 * x + 2) * (4 * x^2) = 28 * x^5 - 20 * x^3 + 8 * x^2 := by
  sorry

end expand_polynomial_l499_49950


namespace consecutive_integers_problem_l499_49944

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = z + 1) →
  (x = z + 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  (z = 2) →
  x = 4 := by
sorry

end consecutive_integers_problem_l499_49944


namespace triangle_perimeter_from_medians_l499_49988

/-- If a triangle has medians of lengths 3, 4, and 6, then its perimeter is 26. -/
theorem triangle_perimeter_from_medians (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (med1 : ∃ (m : ℝ), m = 3 ∧ m^2 = (b^2 + c^2) / 4 - a^2 / 16)
  (med2 : ∃ (m : ℝ), m = 4 ∧ m^2 = (a^2 + c^2) / 4 - b^2 / 16)
  (med3 : ∃ (m : ℝ), m = 6 ∧ m^2 = (a^2 + b^2) / 4 - c^2 / 16) :
  a + b + c = 26 := by
  sorry


end triangle_perimeter_from_medians_l499_49988


namespace line_intercepts_opposite_l499_49974

/-- A line with equation (a-2)x + y - a = 0 has intercepts on the coordinate axes that are opposite numbers if and only if a = 0 or a = 1 -/
theorem line_intercepts_opposite (a : ℝ) : 
  (∃ x y : ℝ, (a - 2) * x + y - a = 0 ∧ 
   ((x = 0 ∧ y ≠ 0) ∨ (x ≠ 0 ∧ y = 0)) ∧
   (x = 0 → y = a) ∧
   (y = 0 → x = a / (a - 2)) ∧
   x = -y) ↔ 
  (a = 0 ∨ a = 1) :=
by sorry

end line_intercepts_opposite_l499_49974


namespace tangent_line_to_logarithmic_curve_l499_49990

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧
    (1 : ℝ) = 1 / (x₀ + a)) →
  a = 2 := by
sorry

end tangent_line_to_logarithmic_curve_l499_49990


namespace max_value_inequality_l499_49973

theorem max_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (6 * a * b) / (9 * b^2 + a^2) + (2 * a * b) / (b^2 + a^2) ≤ 8 / 3 := by
  sorry

end max_value_inequality_l499_49973


namespace four_Y_one_equals_27_l499_49969

/-- Definition of the Y operation -/
def Y (a b : ℝ) : ℝ := 3 * (a^2 - 2*a*b + b^2)

/-- Theorem stating that 4 Y 1 = 27 -/
theorem four_Y_one_equals_27 : Y 4 1 = 27 := by sorry

end four_Y_one_equals_27_l499_49969


namespace quadratic_equation_properties_l499_49936

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 → 
    ((a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
    ((-1 ∈ {x | a * x^2 + b * x + c = 0} ∧ 2 ∈ {x | a * x^2 + b * x + c = 0}) → 2*a + c = 0) ∧
    ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) → 
      ∃ u v, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0))) :=
by
  sorry

end quadratic_equation_properties_l499_49936


namespace geometric_sequence_ratio_l499_49987

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  q : ℝ      -- The common ratio
  h1 : ∀ n, a (n + 1) = a n * q  -- Definition of geometric sequence
  h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)  -- Sum formula for geometric sequence

/-- The theorem statement -/
theorem geometric_sequence_ratio (seq : GeometricSequence) 
  (h3 : seq.a 3 = 4)
  (h4 : seq.S 3 = 12) :
  seq.q = 1 ∨ seq.q = -1/2 :=
sorry

end geometric_sequence_ratio_l499_49987


namespace sin_330_degrees_l499_49914

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_degrees_l499_49914


namespace kasun_family_children_l499_49931

/-- Represents the Kasun family structure and ages -/
structure KasunFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  dog_age : ℕ
  children_total_age : ℕ

/-- The average age of the entire family is 22 years -/
def family_average (f : KasunFamily) : Prop :=
  (f.father_age + f.mother_age + f.children_total_age + f.dog_age) / (2 + f.num_children + 1) = 22

/-- The average age of the mother, children, and the pet dog is 18 years -/
def partial_average (f : KasunFamily) : Prop :=
  (f.mother_age + f.children_total_age + f.dog_age) / (1 + f.num_children + 1) = 18

/-- The theorem stating that the number of children in the Kasun family is 5 -/
theorem kasun_family_children (f : KasunFamily) 
  (h1 : family_average f)
  (h2 : partial_average f)
  (h3 : f.father_age = 50)
  (h4 : f.dog_age = 10) : 
  f.num_children = 5 := by
  sorry

end kasun_family_children_l499_49931


namespace square_of_z_l499_49964

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := 2 + 5 * i

-- Theorem statement
theorem square_of_z : z^2 = -21 + 20 * i := by
  sorry

end square_of_z_l499_49964


namespace expression_value_l499_49975

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4*x*z + y*w = 4) 
  (eq2 : x*w + y*z = 8) : 
  (2*x + y) * (2*z + w) = 20 := by sorry

end expression_value_l499_49975


namespace contract_copies_per_person_l499_49966

theorem contract_copies_per_person 
  (contract_pages : ℕ) 
  (total_pages : ℕ) 
  (num_people : ℕ) 
  (h1 : contract_pages = 20) 
  (h2 : total_pages = 360) 
  (h3 : num_people = 9) :
  (total_pages / contract_pages) / num_people = 2 :=
by
  sorry

end contract_copies_per_person_l499_49966


namespace decision_box_distinguishes_l499_49939

/-- Represents a flowchart element --/
inductive FlowchartElement
  | ProcessingBox
  | DecisionBox
  | InputOutputBox
  | StartEndBox

/-- Represents a flowchart structure --/
structure FlowchartStructure :=
  (elements : Set FlowchartElement)

/-- Definition of a conditional structure --/
def is_conditional (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∈ s.elements ∧ 
  (∃ (b1 b2 : Set FlowchartElement), b1 ⊆ s.elements ∧ b2 ⊆ s.elements ∧ b1 ≠ b2)

/-- Definition of a sequential structure --/
def is_sequential (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∉ s.elements

/-- Theorem: The inclusion of a decision box distinguishes conditional from sequential structures --/
theorem decision_box_distinguishes :
  ∀ (s : FlowchartStructure), 
    (is_conditional s ↔ FlowchartElement.DecisionBox ∈ s.elements) ∧
    (is_sequential s ↔ FlowchartElement.DecisionBox ∉ s.elements) :=
by sorry

end decision_box_distinguishes_l499_49939


namespace liam_chocolate_consumption_l499_49915

/-- Given that Liam ate a total of 150 chocolates in five days, and each day after
    the first day he ate 8 more chocolates than the previous day, prove that
    he ate 38 chocolates on the fourth day. -/
theorem liam_chocolate_consumption :
  ∀ (x : ℕ),
  (x + (x + 8) + (x + 16) + (x + 24) + (x + 32) = 150) →
  (x + 24 = 38) :=
by sorry

end liam_chocolate_consumption_l499_49915


namespace inequalities_proof_l499_49910

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b ≥ 2*a - b) ∧ (a^2 / b + b^2 / c + c^2 / a ≥ a + b + c) := by
  sorry

end inequalities_proof_l499_49910


namespace overtime_pay_ratio_l499_49917

/-- Given Bill's pay structure, prove the ratio of overtime to regular pay rate --/
theorem overtime_pay_ratio (initial_rate : ℝ) (total_pay : ℝ) (total_hours : ℕ) (regular_hours : ℕ) :
  initial_rate = 20 →
  total_pay = 1200 →
  total_hours = 50 →
  regular_hours = 40 →
  (total_pay - initial_rate * regular_hours) / (total_hours - regular_hours) / initial_rate = 2 := by
  sorry

end overtime_pay_ratio_l499_49917


namespace circle_diameter_l499_49933

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end circle_diameter_l499_49933


namespace speed_of_M_constant_l499_49943

/-- Represents a crank-slider mechanism -/
structure CrankSlider where
  ω : ℝ  -- Angular velocity of the crank
  OA : ℝ  -- Length of OA
  AB : ℝ  -- Length of AB
  AM : ℝ  -- Length of AM

/-- The speed of point M in a crank-slider mechanism -/
def speed_of_M (cs : CrankSlider) : ℝ := cs.OA * cs.ω

/-- Theorem: The speed of point M is constant and equal to OA * ω -/
theorem speed_of_M_constant (cs : CrankSlider) 
  (h1 : cs.ω = 10)
  (h2 : cs.OA = 90)
  (h3 : cs.AB = 90)
  (h4 : cs.AM = cs.AB / 2) :
  speed_of_M cs = 900 := by
  sorry

end speed_of_M_constant_l499_49943


namespace solve_bus_problem_l499_49945

def bus_problem (first_stop : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) (third_stop_off : ℕ) (final_count : ℕ) : Prop :=
  let after_first := first_stop
  let after_second := after_first - second_stop_off + second_stop_on
  let before_third_on := after_second - third_stop_off
  ∃ (third_stop_on : ℕ), before_third_on + third_stop_on = final_count ∧ third_stop_on = 4

theorem solve_bus_problem :
  bus_problem 7 3 5 2 11 :=
by
  sorry

#check solve_bus_problem

end solve_bus_problem_l499_49945


namespace arithmetic_sequence_variance_l499_49949

/-- Given an arithmetic sequence with common difference d,
    prove that if the variance of the first five terms is 2, then d = ±1 -/
theorem arithmetic_sequence_variance (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  ((a 1 - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 2*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 3*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 4*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2) / 5 = 2 →
  d = 1 ∨ d = -1 :=
by sorry

end arithmetic_sequence_variance_l499_49949


namespace min_value_of_max_expression_l499_49999

theorem min_value_of_max_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = max (1 / (a * c) + b) (max (1 / a + b * c) (a / b + c)) ∧ M ≥ 2 ∧ 
  (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    max (1 / (a' * c') + b') (max (1 / a' + b' * c') (a' / b' + c')) = 2) :=
by sorry

end min_value_of_max_expression_l499_49999


namespace inequality_holds_l499_49930

theorem inequality_holds (x : ℝ) (h : 0 < x ∧ x < 1) : 0 < 1 - x^2 ∧ 1 - x^2 < 1 := by
  sorry

end inequality_holds_l499_49930


namespace triangle_angle_problem_l499_49925

theorem triangle_angle_problem (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by
  sorry

end triangle_angle_problem_l499_49925


namespace fixed_points_of_f_composition_l499_49971

def f (x : ℝ) : ℝ := x^2 - 5*x + 1

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ 
    x = (5 + Real.sqrt 21) / 2 ∨
    x = (5 - Real.sqrt 21) / 2 ∨
    x = (11 + Real.sqrt 101) / 2 ∨
    x = (11 - Real.sqrt 101) / 2 :=
by sorry

end fixed_points_of_f_composition_l499_49971


namespace lopez_family_seating_arrangements_l499_49997

/-- Represents the number of family members -/
def family_size : ℕ := 5

/-- Represents the number of car seats -/
def car_seats : ℕ := 5

/-- Represents the number of eligible drivers -/
def eligible_drivers : ℕ := 3

/-- Calculates the number of seating arrangements -/
def seating_arrangements (f s d : ℕ) : ℕ :=
  d * (f - 1) * Nat.factorial (f - 2)

/-- Theorem stating the number of seating arrangements for the Lopez family -/
theorem lopez_family_seating_arrangements :
  seating_arrangements family_size car_seats eligible_drivers = 72 :=
by sorry

end lopez_family_seating_arrangements_l499_49997


namespace base8_addition_and_conversion_l499_49924

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 16 --/
def base10_to_base16 (n : ℕ) : ℕ := sorry

/-- Adds two base 8 numbers and returns the result in base 8 --/
def add_base8 (a b : ℕ) : ℕ := 
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b)

theorem base8_addition_and_conversion :
  let a : ℕ := 537 -- In base 8
  let b : ℕ := 246 -- In base 8
  let sum_base8 : ℕ := add_base8 a b
  let sum_base16 : ℕ := base10_to_base16 (base8_to_base10 sum_base8)
  sum_base8 = 1005 ∧ sum_base16 = 0x205 := by sorry

end base8_addition_and_conversion_l499_49924


namespace social_dance_attendance_l499_49934

theorem social_dance_attendance (men : ℕ) (women : ℕ) 
  (men_partners : ℕ) (women_partners : ℕ) :
  men = 15 →
  men_partners = 4 →
  women_partners = 3 →
  men * men_partners = women * women_partners →
  women = 20 := by
sorry

end social_dance_attendance_l499_49934


namespace point_outside_circle_l499_49991

theorem point_outside_circle (a b : ℝ) 
  (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l499_49991


namespace wage_ratio_is_two_to_one_l499_49979

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period -/
def total_earnings (num_workers : ℕ) (num_days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (num_days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one :
  ∃ (men_wage women_wage : ℚ),
    total_earnings 40 10 men_wage = 14400 ∧
    total_earnings 40 30 women_wage = 21600 ∧
    wage_ratio men_wage women_wage = 2 := by
  sorry

end wage_ratio_is_two_to_one_l499_49979


namespace paperclip_production_l499_49900

theorem paperclip_production (machines_base : ℕ) (paperclips_per_minute : ℕ) (machines : ℕ) (minutes : ℕ) :
  machines_base = 8 →
  paperclips_per_minute = 560 →
  machines = 18 →
  minutes = 6 →
  (machines * paperclips_per_minute * minutes) / machines_base = 7560 := by
  sorry

end paperclip_production_l499_49900


namespace lasso_probability_l499_49954

theorem lasso_probability (p : ℝ) (n : ℕ) (hp : p = 1 / 2) (hn : n = 4) :
  1 - (1 - p) ^ n = 15 / 16 :=
by sorry

end lasso_probability_l499_49954


namespace coordinate_sum_of_h_l499_49970

/-- Given a function g where g(2) = 5, and a function h where h(x) = (g(x))^2 for all x,
    the sum of the coordinates of the point (2, h(2)) is 27. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (h_def : ∀ x, h x = (g x)^2) 
    (g_val : g 2 = 5) : 
  2 + h 2 = 27 := by
  sorry

end coordinate_sum_of_h_l499_49970


namespace blackboard_numbers_l499_49938

/-- The sum of reciprocals of the initial numbers on the blackboard -/
def initial_sum (m : ℕ) : ℚ := (2 * m) / (2 * m + 1)

/-- The operation performed in each move -/
def move (a b c : ℚ) : ℚ := (a * b * c) / (a * b + b * c + c * a)

theorem blackboard_numbers (m : ℕ) (h1 : m ≥ 2) :
  ∀ x : ℚ, 
    (∃ (nums : List ℚ), 
      (nums.length = 2) ∧ 
      (4/3 ∈ nums) ∧ 
      (x ∈ nums) ∧ 
      (1 / (4/3) + 1 / x = initial_sum m)) →
    x > 4 := by sorry

end blackboard_numbers_l499_49938


namespace bird_watching_problem_l499_49904

/-- Given 3 bird watchers with an average of 9 birds seen per person,
    where one sees 7 birds and another sees 9 birds,
    prove that the third person must see 11 birds. -/
theorem bird_watching_problem (total_watchers : Nat) (average_birds : Nat) 
  (first_watcher_birds : Nat) (second_watcher_birds : Nat) :
  total_watchers = 3 →
  average_birds = 9 →
  first_watcher_birds = 7 →
  second_watcher_birds = 9 →
  (total_watchers * average_birds - first_watcher_birds - second_watcher_birds) = 11 := by
  sorry

#check bird_watching_problem

end bird_watching_problem_l499_49904


namespace problem_solution_l499_49961

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 72) : x = 9 := by
  sorry

end problem_solution_l499_49961


namespace problem_solution_l499_49968

theorem problem_solution (m n : ℕ) 
  (h1 : m + 10 < n + 1) 
  (h2 : (m + (m + 4) + (m + 10) + (n + 1) + (n + 2) + 2*n) / 6 = n) 
  (h3 : ((m + 10) + (n + 1)) / 2 = n) : 
  m + n = 21 := by
  sorry

end problem_solution_l499_49968


namespace tan_identity_l499_49937

theorem tan_identity (α : ℝ) (h : Real.tan (α + π / 6) = 2) :
  Real.tan (2 * α + 7 * π / 12) = -1 / 7 := by
  sorry

end tan_identity_l499_49937


namespace newOp_seven_three_l499_49941

-- Define the new operation ⊗
def newOp (p q : ℝ) : ℝ := p^2 - 2*q

-- Theorem to prove
theorem newOp_seven_three : newOp 7 3 = 43 := by
  sorry

end newOp_seven_three_l499_49941


namespace floor_equation_solution_l499_49959

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 2⌋ ↔ 5/2 ≤ x ∧ x < 7/2 :=
sorry

end floor_equation_solution_l499_49959


namespace binomial_expansion_coefficient_l499_49983

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ m : ℝ, (1 + 2*x)^3 = 1 + 6*x + m*x^2 + 8*x^3 ∧ m = 12 := by
  sorry

end binomial_expansion_coefficient_l499_49983


namespace ellipse_hyperbola_common_foci_l499_49906

/-- The value of m for which the given ellipse and hyperbola share common foci -/
theorem ellipse_hyperbola_common_foci : ∃ m : ℝ,
  (∀ x y : ℝ, x^2 / 25 + y^2 / 16 = 1 → x^2 / m - y^2 / 5 = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 - b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      ∃ c : ℝ, c^2 = a^2 + b^2 ∧ (x = c ∨ x = -c) ∧ y = 0)) →
  m = 4 :=
sorry

end ellipse_hyperbola_common_foci_l499_49906


namespace log_sqrt10_1000_sqrt10_l499_49960

theorem log_sqrt10_1000_sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end log_sqrt10_1000_sqrt10_l499_49960


namespace sqrt_sum_zero_implies_power_sum_zero_l499_49982

theorem sqrt_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + Real.sqrt (b - 1) = 0 → a^1011 + b^1011 = 0 := by
  sorry

end sqrt_sum_zero_implies_power_sum_zero_l499_49982


namespace divisible_by_55_l499_49976

theorem divisible_by_55 (n : ℤ) : 
  55 ∣ (n^2 + 3*n + 1) ↔ n % 55 = 6 ∨ n % 55 = 46 := by
  sorry

end divisible_by_55_l499_49976


namespace complex_z_value_l499_49903

theorem complex_z_value (z : ℂ) : z / Complex.I = 2 - 3 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end complex_z_value_l499_49903


namespace homework_challenge_l499_49920

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_for_points (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- The homework challenge theorem -/
theorem homework_challenge :
  assignments_for_points 30 = 60 := by
sorry

#eval assignments_for_points 30

end homework_challenge_l499_49920


namespace annas_gold_cost_per_gram_l499_49980

/-- Calculates the cost per gram of Anna's gold -/
theorem annas_gold_cost_per_gram 
  (gary_gold : ℝ) 
  (gary_cost_per_gram : ℝ) 
  (anna_gold : ℝ) 
  (total_cost : ℝ) 
  (h1 : gary_gold = 30)
  (h2 : gary_cost_per_gram = 15)
  (h3 : anna_gold = 50)
  (h4 : total_cost = 1450)
  (h5 : total_cost = gary_gold * gary_cost_per_gram + anna_gold * (total_cost - gary_gold * gary_cost_per_gram) / anna_gold) :
  (total_cost - gary_gold * gary_cost_per_gram) / anna_gold = 20 := by
  sorry

#check annas_gold_cost_per_gram

end annas_gold_cost_per_gram_l499_49980


namespace jimmy_pizza_cost_per_slice_l499_49995

/-- Calculates the cost per slice of a pizza given the following parameters:
    * base_cost: The cost of a large pizza
    * num_slices: The number of slices in a large pizza
    * first_topping_cost: The cost of the first topping
    * next_two_toppings_cost: The cost of each of the next two toppings
    * remaining_toppings_cost: The cost of each remaining topping
    * num_toppings: The total number of toppings ordered
-/
def cost_per_slice (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
                   (next_two_toppings_cost : ℚ) (remaining_toppings_cost : ℚ) 
                   (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost +
                    2 * next_two_toppings_cost +
                    (num_toppings - 3) * remaining_toppings_cost
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  cost_per_slice 10 8 2 1 (1/2) 7 = 2 := by
  sorry

end jimmy_pizza_cost_per_slice_l499_49995


namespace roller_coaster_theorem_l499_49962

/-- The number of different combinations for two rides with 7 people,
    where each ride accommodates 4 people and no person rides more than once. -/
def roller_coaster_combinations : ℕ := 525

/-- The total number of people in the group. -/
def total_people : ℕ := 7

/-- The number of people that can fit in a car for each ride. -/
def people_per_ride : ℕ := 4

/-- The number of rides. -/
def number_of_rides : ℕ := 2

theorem roller_coaster_theorem :
  roller_coaster_combinations =
    (Nat.choose total_people people_per_ride) *
    (Nat.choose (total_people - 1) people_per_ride) :=
by sorry

end roller_coaster_theorem_l499_49962


namespace largest_x_absolute_value_equation_l499_49911

theorem largest_x_absolute_value_equation :
  ∀ x : ℝ, |5 - x| = 15 + x → x ≤ -5 ∧ |-5 - 5| = 15 + (-5) := by
  sorry

end largest_x_absolute_value_equation_l499_49911


namespace function_property_l499_49908

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a
  else Real.log x / Real.log a

-- State the theorem
theorem function_property (a : ℝ) (h : a ≠ 0) (h1 : a ≠ 1) :
  f a (f a 1) = 2 → a = -2 := by sorry

end

end function_property_l499_49908


namespace earring_ratio_l499_49965

theorem earring_ratio (bella_earrings monica_earrings rachel_earrings : ℕ) :
  bella_earrings = 10 ∧
  bella_earrings = monica_earrings / 4 ∧
  bella_earrings + monica_earrings + rachel_earrings = 70 →
  monica_earrings / rachel_earrings = 2 := by
  sorry

end earring_ratio_l499_49965


namespace prime_solution_uniqueness_l499_49984

theorem prime_solution_uniqueness :
  ∀ p q : ℕ,
  Prime p →
  Prime q →
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
  (p = 17 ∧ q = 3) ∨ (p = 3 ∧ q = 17) :=
by sorry

end prime_solution_uniqueness_l499_49984


namespace cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l499_49913

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ cube_set → b ∈ cube_set → (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ a > b ∧ (a - b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b > 0 ∧ (a / b) ∉ cube_set :=
sorry

end cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l499_49913


namespace intersection_inequality_solution_set_l499_49942

/-- Given a line and a hyperbola intersecting at two points, 
    prove the solution set of a related inequality. -/
theorem intersection_inequality_solution_set 
  (k₀ k b m n : ℝ) : 
  (∃ (x : ℝ), k₀ * x + b = k^2 / x ∧ 
              (x = m ∧ k₀ * m + b = -1 ∧ k^2 / m = -1) ∨
              (x = n ∧ k₀ * n + b = 2 ∧ k^2 / n = 2)) →
  {x : ℝ | x^2 > k₀ * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end intersection_inequality_solution_set_l499_49942


namespace volume_of_specific_pyramid_l499_49905

/-- A right pyramid with a square base and an equilateral triangular face --/
structure RightPyramid where
  -- The side length of the equilateral triangular face
  side_length : ℝ
  -- Assumption that the side length is positive
  side_length_pos : side_length > 0

/-- The volume of the right pyramid --/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (500 * Real.sqrt 3) / 3

/-- Theorem stating the volume of the specific pyramid --/
theorem volume_of_specific_pyramid :
  ∃ (p : RightPyramid), p.side_length = 10 ∧ volume p = (500 * Real.sqrt 3) / 3 := by
  sorry

end volume_of_specific_pyramid_l499_49905


namespace team_A_builds_22_5_meters_per_day_l499_49957

def team_A_build_rate : ℝ → Prop := λ x => 
  (150 / x = 100 / (2 * x - 30)) ∧ (x > 0)

theorem team_A_builds_22_5_meters_per_day :
  ∃ x : ℝ, team_A_build_rate x ∧ x = 22.5 := by
  sorry

end team_A_builds_22_5_meters_per_day_l499_49957


namespace f_properties_l499_49935

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| + |x - b|

-- State the theorem
theorem f_properties (a b : ℝ) (h : -1 < a ∧ a < b) :
  -- Part 1
  (∀ x : ℝ, f 1 2 x ≥ Real.sin x) ∧
  -- Part 2
  {x : ℝ | f a b x < a + b + 2} = {x : ℝ | |2*x - a - b| < a + b + 2} :=
by sorry


end f_properties_l499_49935


namespace team_selection_with_quadruplets_l499_49989

/-- The number of ways to choose a team with restrictions on quadruplets -/
def choose_team (total_players : ℕ) (team_size : ℕ) (quadruplets : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - quadruplets) (team_size - quadruplets)

/-- Theorem stating the number of ways to choose the team under given conditions -/
theorem team_selection_with_quadruplets :
  choose_team 16 11 4 = 3576 := by
  sorry

end team_selection_with_quadruplets_l499_49989


namespace inequality_proof_l499_49940

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b * c / a + c * a / b + a * b / c ≥ a + b + c := by
  sorry

end inequality_proof_l499_49940


namespace largest_gold_coins_distribution_l499_49948

theorem largest_gold_coins_distribution (n : ℕ) : 
  n < 100 ∧ 
  n % 13 = 3 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 13 = 3 → m ≤ n) → 
  n = 94 := by
sorry

end largest_gold_coins_distribution_l499_49948


namespace smallest_two_digit_prime_with_composite_reverse_l499_49992

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ 
             Nat.Prime n ∧ 
             tens_digit n = 2 ∧ 
             ¬(Nat.Prime (reverse_digits n)) ∧
             (∀ m : ℕ, is_two_digit m → 
                       Nat.Prime m → 
                       tens_digit m = 2 → 
                       ¬(Nat.Prime (reverse_digits m)) → 
                       n ≤ m) ∧
             n = 23 := by sorry

end smallest_two_digit_prime_with_composite_reverse_l499_49992


namespace two_valid_solutions_exist_l499_49919

def is_valid_solution (a b c d e f g h i : ℕ) : Prop :=
  a ∈ Finset.range 10 ∧ b ∈ Finset.range 10 ∧ c ∈ Finset.range 10 ∧
  d ∈ Finset.range 10 ∧ e ∈ Finset.range 10 ∧ f ∈ Finset.range 10 ∧
  g ∈ Finset.range 10 ∧ h ∈ Finset.range 10 ∧ i ∈ Finset.range 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  (100 * a + 10 * b + c) + d = 10 * e + f ∧
  g * h = 10 * i + f

theorem two_valid_solutions_exist : ∃ (a b c d e f g h i : ℕ),
  is_valid_solution a b c d e f g h i ∧
  ∃ (j k l m n o p q r : ℕ),
  is_valid_solution j k l m n o p q r ∧
  (a ≠ j ∨ b ≠ k ∨ c ≠ l ∨ d ≠ m ∨ e ≠ n ∨ f ≠ o ∨ g ≠ p ∨ h ≠ q ∨ i ≠ r) :=
sorry

end two_valid_solutions_exist_l499_49919


namespace ship_grain_problem_l499_49953

theorem ship_grain_problem (spilled_grain : ℕ) (remaining_grain : ℕ) 
  (h1 : spilled_grain = 49952) (h2 : remaining_grain = 918) : 
  spilled_grain + remaining_grain = 50870 := by
  sorry

end ship_grain_problem_l499_49953


namespace speedster_convertibles_l499_49952

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) 
  (h1 : speedsters = (2 * total) / 3)
  (h2 : convertibles = (4 * speedsters) / 5)
  (h3 : total - speedsters = 60) :
  convertibles = 96 := by
  sorry

end speedster_convertibles_l499_49952


namespace twelfth_term_is_fifteen_l499_49972

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 7 + a 9 = 16
  fourth_term : a 4 = 1

/-- The 12th term of the arithmetic sequence is 15 -/
theorem twelfth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end twelfth_term_is_fifteen_l499_49972


namespace ticket_cost_calculation_l499_49998

/-- Calculates the total amount spent on tickets given the prices and quantities -/
def total_ticket_cost (adult_price child_price : ℚ) (total_tickets child_tickets : ℕ) : ℚ :=
  let adult_tickets := total_tickets - child_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem stating that the total amount spent on tickets is $83.50 -/
theorem ticket_cost_calculation :
  total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16 = (83.50 : ℚ) := by
  sorry

#eval total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16

end ticket_cost_calculation_l499_49998


namespace joe_pays_four_more_than_jenny_l499_49909

/-- Represents the pizza sharing scenario between Jenny and Joe -/
structure PizzaScenario where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  mushroomExtraCost : ℚ
  mushroomSlices : ℕ
  joeMushroomSlices : ℕ
  joePlainSlices : ℕ

/-- Calculates the cost difference between Joe's and Jenny's payments -/
def paymentDifference (scenario : PizzaScenario) : ℚ :=
  let plainSliceCost := scenario.plainPizzaCost / scenario.totalSlices
  let mushroomSliceCost := plainSliceCost + scenario.mushroomExtraCost
  let jennysSlices := scenario.totalSlices - scenario.joeMushroomSlices - scenario.joePlainSlices
  let joePayment := scenario.joeMushroomSlices * mushroomSliceCost + scenario.joePlainSlices * plainSliceCost
  let jennyPayment := jennysSlices * plainSliceCost
  joePayment - jennyPayment

/-- Theorem stating that in the given scenario, Joe pays $4 more than Jenny -/
theorem joe_pays_four_more_than_jenny : 
  let scenario := PizzaScenario.mk 12 12 (1/2) 4 4 3
  paymentDifference scenario = 4 := by sorry

end joe_pays_four_more_than_jenny_l499_49909


namespace line_circle_no_intersection_l499_49994

theorem line_circle_no_intersection (a : ℝ) :
  (∀ x y : ℝ, x + y = a → x^2 + y^2 ≠ 1) ↔ (a > 1 ∨ a < -1) :=
by sorry

end line_circle_no_intersection_l499_49994


namespace promotional_price_equiv_correct_method_l499_49993

/-- Represents the promotional price calculation for books -/
def promotional_price (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Represents the correct method of calculation as described in option C -/
def correct_method (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Theorem stating that the promotional price calculation is equivalent to the correct method -/
theorem promotional_price_equiv_correct_method :
  ∀ x : ℝ, promotional_price x = correct_method x := by
  sorry

end promotional_price_equiv_correct_method_l499_49993


namespace regular_star_polygon_points_l499_49929

/-- A regular star polygon with n points, where each point has two associated angles. -/
structure RegularStarPolygon where
  n : ℕ
  A : Fin n → ℝ
  B : Fin n → ℝ
  all_A_congruent : ∀ i j, A i = A j
  all_B_congruent : ∀ i j, B i = B j
  A_less_than_B : ∀ i, A i = B i - 20

/-- The number of points in a regular star polygon satisfying the given conditions is 18. -/
theorem regular_star_polygon_points (p : RegularStarPolygon) : p.n = 18 := by
  sorry

end regular_star_polygon_points_l499_49929


namespace sum_of_solutions_squared_equation_l499_49978

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 64 ∧ (b - 8)^2 = 64 ∧ a + b = 16) := by
  sorry

end sum_of_solutions_squared_equation_l499_49978


namespace quadratic_function_bound_l499_49926

theorem quadratic_function_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a * x - b * x^2 ≤ 1) → a ≤ 2 * Real.sqrt b := by
  sorry

end quadratic_function_bound_l499_49926


namespace tangent_parallel_to_line_l499_49918

theorem tangent_parallel_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  let tangent_point : ℝ × ℝ := (π / 2, 1)
  let tangent_slope : ℝ := (deriv f) (π / 2)
  let line_slope : ℝ := 1 / a
  (tangent_slope = line_slope) → a = -1 := by
sorry

end tangent_parallel_to_line_l499_49918


namespace optimal_price_for_equipment_l499_49932

/-- Represents the selling price and annual sales volume relationship for a high-tech equipment -/
structure EquipmentSales where
  cost_price : ℝ
  price_volume_1 : ℝ × ℝ
  price_volume_2 : ℝ × ℝ
  max_price : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the equipment -/
def optimal_selling_price (sales : EquipmentSales) : ℝ :=
  sorry

/-- Theorem stating the optimal selling price for the given conditions -/
theorem optimal_price_for_equipment (sales : EquipmentSales)
  (h1 : sales.cost_price = 300000)
  (h2 : sales.price_volume_1 = (350000, 550))
  (h3 : sales.price_volume_2 = (400000, 500))
  (h4 : sales.max_price = 600000)
  (h5 : sales.target_profit = 80000000) :
  optimal_selling_price sales = 500000 :=
sorry

end optimal_price_for_equipment_l499_49932


namespace concert_ticket_sales_l499_49963

/-- Proves that given the conditions of the concert ticket sales, the number of back seat tickets sold is 14,500 --/
theorem concert_ticket_sales 
  (total_seats : ℕ) 
  (main_seat_price back_seat_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 20000)
  (h2 : main_seat_price = 55)
  (h3 : back_seat_price = 45)
  (h4 : total_revenue = 955000) :
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_seat_price * main_seats + back_seat_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

end concert_ticket_sales_l499_49963


namespace interval_equivalence_l499_49956

theorem interval_equivalence (x : ℝ) : 
  (3/4 < x ∧ x < 4/5) ↔ (3 < 5*x + 1 ∧ 5*x + 1 < 5) ∧ (3 < 4*x ∧ 4*x < 5) :=
by sorry

end interval_equivalence_l499_49956


namespace equidistant_points_bound_l499_49902

/-- A set of points in a plane where no three points are collinear -/
structure PointSet where
  S : Set (ℝ × ℝ)
  noncollinear : ∀ (p q r : ℝ × ℝ), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r →
    (p.1 - q.1) * (r.2 - q.2) ≠ (r.1 - q.1) * (p.2 - q.2)

/-- The property that for each point, there are k equidistant points -/
def has_k_equidistant (PS : PointSet) (k : ℕ) : Prop :=
  ∀ p ∈ PS.S, ∃ (T : Set (ℝ × ℝ)), T ⊆ PS.S ∧ T.ncard = k ∧
    ∀ q ∈ T, q ≠ p → ∃ d : ℝ, d > 0 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = d^2

theorem equidistant_points_bound (n k : ℕ) (h_pos : 0 < n ∧ 0 < k) (PS : PointSet)
    (h_card : PS.S.ncard = n) (h_equi : has_k_equidistant PS k) :
    k ≤ (1 : ℝ)/2 + Real.sqrt (2 * n) := by
  sorry

end equidistant_points_bound_l499_49902


namespace sin_cos_pi_12_l499_49901

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l499_49901


namespace sum_of_special_primes_is_prime_l499_49996

theorem sum_of_special_primes_is_prime (P Q : ℕ) : 
  P > 0 ∧ Q > 0 ∧ 
  Nat.Prime P ∧ Nat.Prime Q ∧ Nat.Prime (P - Q) ∧ Nat.Prime (P + Q) →
  Nat.Prime (P + Q + (P - Q) + P + Q) :=
by sorry

end sum_of_special_primes_is_prime_l499_49996


namespace range_of_a_l499_49985

/-- The line passing through points on a 2D plane. -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if two points are on opposite sides of a line. -/
def oppositeSides (p1 p2 : Point2D) (l : Line2D) : Prop :=
  (l.a * p1.x + l.b * p1.y + l.c) * (l.a * p2.x + l.b * p2.y + l.c) < 0

/-- The theorem statement. -/
theorem range_of_a :
  ∀ a : ℝ,
  (oppositeSides (Point2D.mk 0 0) (Point2D.mk 1 1) (Line2D.mk 1 1 (-a))) ↔
  (0 < a ∧ a < 2) :=
sorry

end range_of_a_l499_49985


namespace choose_20_6_l499_49986

theorem choose_20_6 : Nat.choose 20 6 = 2584 := by
  sorry

end choose_20_6_l499_49986


namespace shipment_total_correct_l499_49916

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The percentage of novels displayed in the window -/
def display_percentage : ℚ := 30 / 100

/-- The number of novels left in the stockroom -/
def stockroom_novels : ℕ := 210

/-- Theorem stating that the total number of novels is correct given the conditions -/
theorem shipment_total_correct :
  (1 - display_percentage) * total_novels = stockroom_novels := by
  sorry

end shipment_total_correct_l499_49916


namespace least_subtraction_for_divisibility_problem_solution_l499_49927

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) : 
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution : 
  ∃ (k : ℕ), k = 3 ∧ (427398 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (427398 - m) % 15 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l499_49927


namespace ball_count_and_probability_l499_49946

/-- Represents the colors of the balls -/
inductive Color
  | Red
  | White
  | Blue

/-- Represents the bag of balls -/
structure Bag where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the second bag with specific balls -/
structure SpecificBag where
  red1 : Bool
  white1 : Bool
  blue2 : Bool
  blue3 : Bool

def Bag.probability (b : Bag) (c : Color) : Rat :=
  match c with
  | Color.Red => b.red / b.total
  | Color.White => b.white / b.total
  | Color.Blue => b.blue / b.total

theorem ball_count_and_probability (b : Bag) :
  b.total = 24 ∧ b.blue = 3 ∧ b.probability Color.Red = 1/6 →
  b.red = 4 ∧
  (let sb : SpecificBag := ⟨true, true, true, true⟩
   (5 : Rat) / 12 = (Nat.choose 3 1 * Nat.choose 1 1) / (Nat.choose 4 2)) := by
  sorry


end ball_count_and_probability_l499_49946


namespace g_of_negative_three_l499_49977

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_negative_three : g (-3) = -17 := by
  sorry

end g_of_negative_three_l499_49977


namespace smallest_n_with_triple_sum_l499_49921

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: The smallest positive integer N whose sum of digits is three times 
    the sum of digits of N+1 has a sum of digits equal to 12 -/
theorem smallest_n_with_triple_sum : 
  ∃ (N : ℕ), N > 0 ∧ 
  sum_of_digits N = 3 * sum_of_digits (N + 1) ∧
  sum_of_digits N = 12 ∧
  ∀ (M : ℕ), M > 0 → sum_of_digits M = 3 * sum_of_digits (M + 1) → 
    sum_of_digits M ≥ 12 :=
sorry

end smallest_n_with_triple_sum_l499_49921


namespace lcm_hcf_relation_l499_49923

theorem lcm_hcf_relation (a b : ℕ) (h : a = 24 ∧ b = 198) :
  Nat.lcm a b = 792 :=
by
  sorry

#check lcm_hcf_relation

end lcm_hcf_relation_l499_49923


namespace workshop_prize_difference_l499_49912

theorem workshop_prize_difference (total : ℕ) (wolf : ℕ) (both : ℕ) (nobel : ℕ) 
  (h_total : total = 50)
  (h_wolf : wolf = 31)
  (h_both : both = 14)
  (h_nobel : nobel = 25)
  (h_wolf_less : wolf ≤ total)
  (h_both_less : both ≤ wolf)
  (h_both_less_nobel : both ≤ nobel)
  (h_nobel_less : nobel ≤ total) :
  let non_wolf := total - wolf
  let nobel_non_wolf := nobel - both
  let non_nobel_non_wolf := non_wolf - nobel_non_wolf
  nobel_non_wolf - non_nobel_non_wolf = 3 := by
  sorry

end workshop_prize_difference_l499_49912


namespace composite_product_division_l499_49981

def first_six_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12
def next_six_composite_product : ℕ := 14 * 15 * 16 * 18 * 20 * 21

theorem composite_product_division :
  (first_six_composite_product : ℚ) / next_six_composite_product = 1 / 49 := by
  sorry

end composite_product_division_l499_49981


namespace sequence_sum_l499_49947

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 9*x₂ + 25*x₃ + 49*x₄ + 81*x₅ + 121*x₆ + 169*x₇ = 2)
  (eq2 : 9*x₁ + 25*x₂ + 49*x₃ + 81*x₄ + 121*x₅ + 169*x₆ + 225*x₇ = 24)
  (eq3 : 25*x₁ + 49*x₂ + 81*x₃ + 121*x₄ + 169*x₅ + 225*x₆ + 289*x₇ = 246) :
  49*x₁ + 81*x₂ + 121*x₃ + 169*x₄ + 225*x₅ + 289*x₆ + 361*x₇ = 668 := by
  sorry

end sequence_sum_l499_49947
