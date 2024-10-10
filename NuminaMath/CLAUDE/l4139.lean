import Mathlib

namespace elimination_method_l4139_413916

theorem elimination_method (x y : ℝ) : 
  (5 * x - 2 * y = 4) → 
  (2 * x + 3 * y = 9) → 
  ∃ (a b : ℝ), a = 2 ∧ b = -5 ∧ 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = a * 4 + b * 9) ∧
  (a * 5 + b * 2 = 0) :=
sorry

end elimination_method_l4139_413916


namespace statue_original_cost_l4139_413925

/-- If a statue is sold for $660 with a 20% profit, then its original cost was $550. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 660 → profit_percentage = 0.20 → 
  selling_price = (1 + profit_percentage) * 550 := by
sorry

end statue_original_cost_l4139_413925


namespace exists_log_sum_eq_log_sum_skew_lines_iff_no_common_plane_l4139_413903

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Proposition p
theorem exists_log_sum_eq_log_sum : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ log (a + b) = log a + log b :=
sorry

-- Define a type for lines in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define what it means for a line to lie on a plane
def line_on_plane (l : Line3D) (p : Plane3D) : Prop :=
sorry

-- Define what it means for two lines to be skew
def skew_lines (l1 l2 : Line3D) : Prop :=
∀ (p : Plane3D), ¬(line_on_plane l1 p ∧ line_on_plane l2 p)

-- Proposition q
theorem skew_lines_iff_no_common_plane (l1 l2 : Line3D) :
  skew_lines l1 l2 ↔ ∀ (p : Plane3D), ¬(line_on_plane l1 p ∧ line_on_plane l2 p) :=
sorry

end exists_log_sum_eq_log_sum_skew_lines_iff_no_common_plane_l4139_413903


namespace expression_evaluation_l4139_413926

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1) :
  2 * x^2 - (2*x*y - 3*y^2) + 2*(x^2 + x*y - 2*y^2) = 15 := by
  sorry

end expression_evaluation_l4139_413926


namespace sunflower_seeds_count_l4139_413923

/-- The number of sunflower plants -/
def num_sunflowers : ℕ := 6

/-- The number of dandelion plants -/
def num_dandelions : ℕ := 8

/-- The number of seeds per dandelion plant -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of total seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64/100

/-- The number of seeds per sunflower plant -/
def seeds_per_sunflower : ℕ := 9

theorem sunflower_seeds_count :
  let total_dandelion_seeds := num_dandelions * seeds_per_dandelion
  let total_seeds := total_dandelion_seeds / dandelion_seed_percentage
  let total_sunflower_seeds := total_seeds - total_dandelion_seeds
  seeds_per_sunflower = total_sunflower_seeds / num_sunflowers := by
sorry

end sunflower_seeds_count_l4139_413923


namespace max_classes_less_than_1968_l4139_413902

/-- Relation between two natural numbers where they belong to the same class if one can be obtained from the other by deleting two adjacent digits or identical groups of digits -/
def SameClass (m n : ℕ) : Prop := sorry

/-- The maximum number of equivalence classes under the SameClass relation -/
def MaxClasses : ℕ := sorry

theorem max_classes_less_than_1968 : MaxClasses < 1968 := by sorry

end max_classes_less_than_1968_l4139_413902


namespace quadratic_necessary_not_sufficient_l4139_413921

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, (|x - 2| < 1) → (x^2 - 5*x + 4 < 0)) ∧
  (∃ x : ℝ, (x^2 - 5*x + 4 < 0) ∧ ¬(|x - 2| < 1)) :=
by sorry

end quadratic_necessary_not_sufficient_l4139_413921


namespace sum_inequality_l4139_413914

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11) ≤ 1 / 4 :=
sorry

end sum_inequality_l4139_413914


namespace problem_statement_l4139_413913

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end problem_statement_l4139_413913


namespace complex_modulus_l4139_413932

theorem complex_modulus (x y : ℝ) (z : ℂ) (h : z = x + y * I) 
  (eq : (1/2 * x - y) + (x + y) * I = 3 * I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l4139_413932


namespace ball_probabilities_l4139_413905

/-- The probability of drawing a red ball on the second draw -/
def prob_red_second (total : ℕ) (red : ℕ) : ℚ :=
  red / total

/-- The probability of drawing two balls of the same color -/
def prob_same_color (total : ℕ) (red : ℕ) (green : ℕ) : ℚ :=
  (red * (red - 1) + green * (green - 1)) / (total * (total - 1))

/-- The probability of drawing two red balls -/
def prob_two_red (total : ℕ) (red : ℕ) : ℚ :=
  (red * (red - 1)) / (total * (total - 1))

theorem ball_probabilities :
  let total := 6
  let red := 2
  let green := 4
  (prob_red_second total red = 1/3) ∧
  (prob_same_color total red green = 7/15) ∧
  (∃ n : ℕ, prob_two_red (n + 2) 2 = 1/21 ∧ n = 5) :=
by sorry


end ball_probabilities_l4139_413905


namespace scientific_notation_152300_l4139_413908

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_152300 :
  toScientificNotation 152300 = ScientificNotation.mk 1.523 5 (by norm_num) :=
sorry

end scientific_notation_152300_l4139_413908


namespace dodecahedron_edge_probability_l4139_413930

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_selection_probability d = 3 / 19 := by
  sorry

end dodecahedron_edge_probability_l4139_413930


namespace complex_fraction_simplification_l4139_413901

theorem complex_fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by sorry

end complex_fraction_simplification_l4139_413901


namespace fixed_point_and_bisecting_line_l4139_413911

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

-- Define lines l₁ and l₂
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    ∃ t : ℝ, 
      line_l1 t (-4*t-3) ∧ 
      line_l2 (-t-2) (4*t+7) ∧
      point_P = ((t + (-t-2))/2, ((-4*t-3) + (4*t+7))/2)) :=
sorry

end fixed_point_and_bisecting_line_l4139_413911


namespace boxes_with_pans_is_eight_l4139_413929

/-- Represents the arrangement of teacups and boxes. -/
structure TeacupArrangement where
  total_boxes : Nat
  cups_per_box : Nat
  cups_broken_per_box : Nat
  cups_left : Nat

/-- Calculates the number of boxes containing pans. -/
def boxes_with_pans (arrangement : TeacupArrangement) : Nat :=
  let teacup_boxes := arrangement.cups_left / (arrangement.cups_per_box - arrangement.cups_broken_per_box)
  let remaining_boxes := arrangement.total_boxes - teacup_boxes
  remaining_boxes / 2

/-- Theorem stating that the number of boxes with pans is 8. -/
theorem boxes_with_pans_is_eight : 
  boxes_with_pans { total_boxes := 26
                  , cups_per_box := 20
                  , cups_broken_per_box := 2
                  , cups_left := 180 } = 8 := by
  sorry


end boxes_with_pans_is_eight_l4139_413929


namespace hungarian_olympiad_1959_l4139_413922

theorem hungarian_olympiad_1959 (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  ∃ k : ℤ, (x^n * (y-z) + y^n * (z-x) + z^n * (x-y)) / ((x-y)*(x-z)*(y-z)) = k :=
sorry

end hungarian_olympiad_1959_l4139_413922


namespace arithmetic_operations_with_five_l4139_413924

theorem arithmetic_operations_with_five (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 → x = 1 := by
  sorry

end arithmetic_operations_with_five_l4139_413924


namespace backpack_price_calculation_l4139_413935

theorem backpack_price_calculation
  (num_backpacks : ℕ)
  (monogram_cost : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : monogram_cost = 12)
  (h3 : total_cost = 140) :
  (total_cost - num_backpacks * monogram_cost) / num_backpacks = 16 :=
by sorry

end backpack_price_calculation_l4139_413935


namespace unique_point_on_line_l4139_413934

-- Define the line passing through (4, 11) and (16, 1)
def line_equation (x y : ℤ) : Prop :=
  5 * x + 6 * y = 43

-- Define the condition for positive integers
def positive_integer (n : ℤ) : Prop :=
  0 < n

theorem unique_point_on_line :
  ∃! p : ℤ × ℤ, line_equation p.1 p.2 ∧ positive_integer p.1 ∧ positive_integer p.2 ∧ p = (5, 3) :=
by
  sorry

#check unique_point_on_line

end unique_point_on_line_l4139_413934


namespace line_xz_plane_intersection_l4139_413928

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := { p1 := ⟨2, -1, 3⟩, p2 := ⟨6, 7, -2⟩ }
  let p : Point3D := ⟨2.5, 0, 2.375⟩
  pointOnLine p l ∧ p ∈ xzPlane :=
by sorry

end line_xz_plane_intersection_l4139_413928


namespace hyperbola_chord_of_contact_l4139_413918

/-- The equation of the chord of contact for a hyperbola -/
theorem hyperbola_chord_of_contact 
  (a b x₀ y₀ : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_not_on_hyperbola : (x₀^2 / a^2) - (y₀^2 / b^2) ≠ 1) :
  ∃ (P₁ P₂ : ℝ × ℝ),
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → 
      ((x₀ * x / a^2) - (y₀ * y / b^2) = 1 ↔ 
        (∃ t : ℝ, (x, y) = t • P₁ + (1 - t) • P₂))) :=
sorry

end hyperbola_chord_of_contact_l4139_413918


namespace base6_addition_l4139_413915

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The first number in base 6 -/
def num1 : List Nat := [2, 3, 4, 3]

/-- The second number in base 6 -/
def num2 : List Nat := [1, 5, 3, 2, 5]

/-- The expected result in base 6 -/
def result : List Nat := [2, 2, 1, 1, 2]

theorem base6_addition :
  decimalToBase6 (base6ToDecimal num1 + base6ToDecimal num2) = result := by
  sorry

end base6_addition_l4139_413915


namespace recipe_total_cups_l4139_413917

/-- Given a recipe with a ratio of butter:flour:sugar as 2:5:3 and using 9 cups of sugar,
    the total amount of ingredients used is 30 cups. -/
theorem recipe_total_cups (butter flour sugar total : ℚ) : 
  butter / sugar = 2 / 3 →
  flour / sugar = 5 / 3 →
  sugar = 9 →
  total = butter + flour + sugar →
  total = 30 := by
sorry

end recipe_total_cups_l4139_413917


namespace tablet_interval_l4139_413909

/-- Given a person who takes 5 tablets over 60 minutes with equal intervals, 
    prove that the interval between tablets is 15 minutes. -/
theorem tablet_interval (total_tablets : ℕ) (total_time : ℕ) (h1 : total_tablets = 5) (h2 : total_time = 60) :
  (total_time / (total_tablets - 1) : ℚ) = 15 := by
  sorry

end tablet_interval_l4139_413909


namespace football_games_total_cost_l4139_413906

/-- Represents the attendance and cost data for a month of football games --/
structure MonthData where
  games : ℕ
  ticketCost : ℕ

/-- Calculates the total spent for a given month --/
def monthlyTotal (md : MonthData) : ℕ := md.games * md.ticketCost

/-- The problem statement --/
theorem football_games_total_cost 
  (thisMonth : MonthData)
  (lastMonth : MonthData)
  (nextMonth : MonthData)
  (h1 : thisMonth = { games := 11, ticketCost := 25 })
  (h2 : lastMonth = { games := 17, ticketCost := 30 })
  (h3 : nextMonth = { games := 16, ticketCost := 35 }) :
  monthlyTotal thisMonth + monthlyTotal lastMonth + monthlyTotal nextMonth = 1345 := by
  sorry

end football_games_total_cost_l4139_413906


namespace midpoint_trajectory_l4139_413919

/-- The trajectory of the midpoint between a point on a parabola and a fixed point -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) : 
  y₁ = 2 * x₁^2 + 1 →  -- P is on the parabola y = 2x^2 + 1
  x = (x₁ + 0) / 2 →   -- x-coordinate of midpoint M
  y = (y₁ + (-1)) / 2 → -- y-coordinate of midpoint M
  y = 4 * x^2 :=        -- trajectory equation of M
by sorry

end midpoint_trajectory_l4139_413919


namespace gcd_of_16434_24651_43002_l4139_413900

theorem gcd_of_16434_24651_43002 : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end gcd_of_16434_24651_43002_l4139_413900


namespace min_prize_cost_is_11_l4139_413904

def min_prize_cost (x y : ℕ) : ℕ := 3 * x + 2 * y

theorem min_prize_cost_is_11 :
  ∃ (x y : ℕ),
    x + y ≤ 10 ∧
    (x : ℤ) - y ≤ 2 ∧
    y - x ≤ 2 ∧
    x ≥ 3 ∧
    min_prize_cost x y = 11 ∧
    ∀ (a b : ℕ), a + b ≤ 10 → (a : ℤ) - b ≤ 2 → b - a ≤ 2 → a ≥ 3 → min_prize_cost a b ≥ 11 :=
by
  sorry

end min_prize_cost_is_11_l4139_413904


namespace gcd_7200_13230_l4139_413933

theorem gcd_7200_13230 : Int.gcd 7200 13230 = 30 := by
  sorry

end gcd_7200_13230_l4139_413933


namespace characterize_satisfying_functions_l4139_413931

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f (x^u * y^v) ≤ f x^(1/(4*u)) * f y^(1/(4*v))

/-- The main theorem stating the form of functions satisfying the condition -/
theorem characterize_satisfying_functions :
  ∀ (f : ℝ → ℝ), (∀ x, x > 1 → f x > 1) →
    SatisfiesCondition f →
    ∃ (c : ℝ), c > 1 ∧ ∀ x, x > 1 → f x = c^(1/Real.log x) :=
by sorry

end characterize_satisfying_functions_l4139_413931


namespace double_inequality_solution_l4139_413910

theorem double_inequality_solution (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 2*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 2*x + 5) < 1) ↔
  (5/7 < x ∧ x < 5/3) ∨ (5 < x) :=
by sorry

end double_inequality_solution_l4139_413910


namespace game_result_l4139_413920

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 702 := by
  sorry

end game_result_l4139_413920


namespace apple_purchase_multiple_l4139_413912

theorem apple_purchase_multiple : ∀ x : ℕ,
  (15 : ℕ) + 15 * x + 60 * x = (240 : ℕ) → x = 3 := by
  sorry

end apple_purchase_multiple_l4139_413912


namespace negation_of_proposition_l4139_413907

theorem negation_of_proposition (p : Prop) : 
  (p = ∀ x : ℝ, 2 * x^2 + 1 > 0) → 
  (¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end negation_of_proposition_l4139_413907


namespace theatre_sales_calculation_l4139_413927

/-- Calculates the total sales amount for a theatre performance given ticket prices and quantities sold. -/
theorem theatre_sales_calculation 
  (price1 price2 : ℚ) 
  (total_tickets sold1 : ℕ) 
  (h1 : price1 = 4.5)
  (h2 : price2 = 6)
  (h3 : total_tickets = 380)
  (h4 : sold1 = 205) :
  price1 * sold1 + price2 * (total_tickets - sold1) = 1972.5 :=
by sorry

end theatre_sales_calculation_l4139_413927
