import Mathlib

namespace max_value_inequality_l1438_143863

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -4)
  (abc_nonneg : a * b * c ≥ 0) :
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 16) ≤ 2 * Real.sqrt 30 :=
by sorry

end max_value_inequality_l1438_143863


namespace earth_rotation_certain_l1438_143816

-- Define the type for events
inductive Event : Type
  | EarthRotation : Event
  | RainTomorrow : Event
  | TimeBackwards : Event
  | SnowfallWinter : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.EarthRotation => True
  | _ => False

-- Define the conditions given in the problem
axiom earth_rotation_continuous : ∀ (t : ℝ), ∃ (angle : ℝ), angle ≥ 0 ∧ angle < 360
axiom weather_probabilistic : ∃ (p : ℝ), 0 < p ∧ p < 1
axiom time_forwards : ∀ (t1 t2 : ℝ), t1 < t2 → t1 ≠ t2
axiom snowfall_not_guaranteed : ∃ (winter : Set ℝ), ∃ (day : ℝ), day ∈ winter ∧ ¬∃ (snow : ℝ), snow > 0

-- The theorem to prove
theorem earth_rotation_certain : is_certain Event.EarthRotation :=
  sorry

end earth_rotation_certain_l1438_143816


namespace problem_solution_l1438_143825

theorem problem_solution (x y z : ℝ) :
  (1.5 * x = 0.3 * y) →
  (x = 20) →
  (0.6 * y = z) →
  z = 60 := by
sorry

end problem_solution_l1438_143825


namespace unique_number_sum_of_digits_l1438_143837

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem unique_number_sum_of_digits :
  ∃! N : ℕ, 
    400 < N ∧ N < 600 ∧ 
    N % 2 = 1 ∧ 
    N % 5 = 0 ∧ 
    N % 11 = 0 ∧
    sum_of_digits N = 18 := by
  sorry

end unique_number_sum_of_digits_l1438_143837


namespace spherical_to_cartesian_l1438_143838

/-- Conversion from spherical coordinates to Cartesian coordinates -/
theorem spherical_to_cartesian :
  let r : ℝ := 8
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x, y, z) = (6, 2 * Real.sqrt 3, 4) :=
by sorry

end spherical_to_cartesian_l1438_143838


namespace sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l1438_143894

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem sufficient_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  (planeParallel α β ∧ lineInPlane m β) → lineParallelPlane m α :=
sorry

-- State that the condition is not necessary
theorem not_necessary_condition_for_line_parallel_plane 
  (α β : Plane) (m : Line) :
  ¬(lineParallelPlane m α → (planeParallel α β ∧ lineInPlane m β)) :=
sorry

end sufficient_condition_for_line_parallel_plane_not_necessary_condition_for_line_parallel_plane_l1438_143894


namespace sequence_property_main_theorem_l1438_143819

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2)

def sequence_S (n : ℕ) : ℚ := (4 / 3) ^ (n - 1)

theorem sequence_property : ∀ n : ℕ, n ≥ 1 → 3 * sequence_a (n + 1) = sequence_S n :=
  sorry

theorem main_theorem : ∀ n : ℕ, n ≥ 1 → 
  sequence_a n = if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2) :=
  sorry

end sequence_property_main_theorem_l1438_143819


namespace expand_product_l1438_143822

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l1438_143822


namespace complex_numbers_count_is_25_l1438_143891

def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

def complex_numbers_count : ℕ :=
  (S.filter (λ b => b ≠ 0)).card * (S.card - 1)

theorem complex_numbers_count_is_25 : complex_numbers_count = 25 := by
  sorry

end complex_numbers_count_is_25_l1438_143891


namespace cube_sum_over_product_l1438_143839

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 20)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 13 := by
  sorry

end cube_sum_over_product_l1438_143839


namespace sum_of_squared_differences_to_fourth_power_l1438_143892

theorem sum_of_squared_differences_to_fourth_power :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 := by
  sorry

end sum_of_squared_differences_to_fourth_power_l1438_143892


namespace connie_calculation_l1438_143834

theorem connie_calculation (x : ℤ) : x + 2 = 80 → x - 2 = 76 := by
  sorry

end connie_calculation_l1438_143834


namespace chair_rows_theorem_l1438_143884

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem stating that for 432 total chairs and 16 chairs per row, there are 27 rows -/
theorem chair_rows_theorem :
  calculate_rows 432 16 = 27 := by
  sorry

end chair_rows_theorem_l1438_143884


namespace not_perfect_square_l1438_143835

-- Define a function to create a number with n ones
def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

-- Define our specific number N
def N (k : ℕ) : ℕ := 
  ones 300 * 10^k

-- Theorem statement
theorem not_perfect_square (k : ℕ) : 
  ¬ ∃ (m : ℕ), N k = m^2 := by
  sorry

end not_perfect_square_l1438_143835


namespace speed_ratio_l1438_143869

/-- Two people walk in opposite directions for 1 hour and swap destinations -/
structure WalkProblem where
  /-- Speed of person A in km/h -/
  v₁ : ℝ
  /-- Speed of person B in km/h -/
  v₂ : ℝ
  /-- Both speeds are positive -/
  h₁ : v₁ > 0
  h₂ : v₂ > 0
  /-- Person A reaches B's destination 35 minutes after B reaches A's destination -/
  h₃ : v₂ / v₁ - v₁ / v₂ = 35 / 60

/-- The ratio of speeds is 3:4 -/
theorem speed_ratio (w : WalkProblem) : w.v₁ / w.v₂ = 3 / 4 := by
  sorry

end speed_ratio_l1438_143869


namespace trapezoid_properties_l1438_143861

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  side1 : ℝ
  side2 : ℝ
  diagonal_is_bisector : Bool

/-- Properties of the specific trapezoid in the problem -/
def problem_trapezoid : IsoscelesTrapezoid :=
  { side1 := 6
  , side2 := 6.25
  , diagonal_is_bisector := true }

/-- The length of the diagonal from the acute angle vertex -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

/-- The area of the trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the properties of the specific trapezoid -/
theorem trapezoid_properties :
  let t := problem_trapezoid
  abs (diagonal_length t - 10.423) < 0.001 ∧
  abs (trapezoid_area t - 32) < 0.001 := by sorry

end trapezoid_properties_l1438_143861


namespace cube_root_of_decimal_l1438_143875

theorem cube_root_of_decimal (x : ℚ) : x = 1/4 → x^3 = 15625/1000000 := by
  sorry

end cube_root_of_decimal_l1438_143875


namespace pages_per_day_to_finish_on_time_l1438_143887

/-- Given a 66-page paper due in 6 days, prove that 11 pages per day are required to finish on time. -/
theorem pages_per_day_to_finish_on_time :
  let total_pages : ℕ := 66
  let days_until_due : ℕ := 6
  let pages_per_day : ℕ := total_pages / days_until_due
  pages_per_day = 11 := by sorry

end pages_per_day_to_finish_on_time_l1438_143887


namespace distribute_three_books_twelve_students_l1438_143821

/-- The number of ways to distribute n identical objects among k people,
    where no person can receive more than one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose k n

theorem distribute_three_books_twelve_students :
  distribute 3 12 = 220 := by
  sorry

end distribute_three_books_twelve_students_l1438_143821


namespace decision_block_two_exits_other_blocks_not_two_exits_l1438_143826

/-- Enumeration of program block types -/
inductive ProgramBlock
  | Output
  | Processing
  | Decision
  | StartEnd

/-- Function to determine the number of exits for each program block -/
def num_exits (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.Output => 1
  | ProgramBlock.Processing => 1
  | ProgramBlock.Decision => 2
  | ProgramBlock.StartEnd => 0

/-- Theorem stating that only the Decision block has two exits -/
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock), num_exits block = 2 ↔ block = ProgramBlock.Decision :=
by sorry

/-- Corollary: No other block type has two exits -/
theorem other_blocks_not_two_exits :
  ∀ (block : ProgramBlock), block ≠ ProgramBlock.Decision → num_exits block ≠ 2 :=
by sorry

end decision_block_two_exits_other_blocks_not_two_exits_l1438_143826


namespace club_membership_l1438_143846

theorem club_membership (total : ℕ) (difference : ℕ) (first_year : ℕ) : 
  total = 128 →
  difference = 12 →
  first_year = total / 2 + difference / 2 →
  first_year = 70 :=
by
  sorry

end club_membership_l1438_143846


namespace pointDifference_l1438_143805

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team based on their performance --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  team.wins * 3 + team.draws * 1

/-- The scoring system and match results for Joe's team and the first-place team --/
def joesTeam : TeamPerformance := ⟨1, 3⟩
def firstPlaceTeam : TeamPerformance := ⟨2, 2⟩

/-- The theorem stating the difference in points between the first-place team and Joe's team --/
theorem pointDifference : calculatePoints firstPlaceTeam - calculatePoints joesTeam = 2 := by
  sorry

end pointDifference_l1438_143805


namespace smallest_three_digit_sum_of_powers_l1438_143852

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The main theorem statement -/
theorem smallest_three_digit_sum_of_powers :
  ∃ (K a b : ℕ), 
    isThreeDigit K ∧
    isOneDigitPositive a ∧
    isOneDigitPositive b ∧
    K = a^b + b^a ∧
    (∀ (K' a' b' : ℕ), 
      isThreeDigit K' ∧ 
      isOneDigitPositive a' ∧ 
      isOneDigitPositive b' ∧ 
      K' = a'^b' + b'^a' → 
      K ≤ K') ∧
    K = 100 :=
by sorry

end smallest_three_digit_sum_of_powers_l1438_143852


namespace election_percentage_l1438_143854

theorem election_percentage (total_votes : ℕ) (vote_difference : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  vote_difference = 1650 →
  candidate_percentage = 35 / 100 →
  (candidate_percentage * total_votes : ℚ) + 
  (candidate_percentage * total_votes : ℚ) + vote_difference = total_votes :=
by sorry

end election_percentage_l1438_143854


namespace at_least_one_not_less_than_two_l1438_143878

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / a ≥ 2) :=
by sorry

end at_least_one_not_less_than_two_l1438_143878


namespace theater_ticket_sales_l1438_143813

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Represents the conditions of the theater ticket sales --/
structure TicketSalesConditions where
  tickets : TheaterTickets
  totalRevenue : ℕ
  balconyExcess : ℕ

/-- The theorem to be proved --/
theorem theater_ticket_sales 
  (conditions : TicketSalesConditions) 
  (h1 : totalRevenue conditions.tickets = conditions.totalRevenue)
  (h2 : conditions.tickets.balcony = conditions.tickets.orchestra + conditions.balconyExcess)
  (h3 : conditions.totalRevenue = 3320)
  (h4 : conditions.balconyExcess = 115) :
  conditions.tickets.orchestra + conditions.tickets.balcony = 355 := by
  sorry

#check theater_ticket_sales

end theater_ticket_sales_l1438_143813


namespace floor_abs_negative_real_l1438_143853

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by
  sorry

end floor_abs_negative_real_l1438_143853


namespace factor_expression_l1438_143880

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) := by
  sorry

end factor_expression_l1438_143880


namespace smallest_base_for_61_digits_l1438_143883

theorem smallest_base_for_61_digits : ∃ (b : ℕ), b > 1 ∧ 
  (∀ (n : ℕ), n > 1 → n < b → (Nat.log 10 (n^200) + 1 < 61)) ∧ 
  (Nat.log 10 (b^200) + 1 = 61) := by
  sorry

end smallest_base_for_61_digits_l1438_143883


namespace nested_sqrt_fourth_power_l1438_143823

theorem nested_sqrt_fourth_power : 
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end nested_sqrt_fourth_power_l1438_143823


namespace unique_triangle_with_perimeter_8_l1438_143800

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an integer triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

theorem unique_triangle_with_perimeter_8 :
  ∃! (t : IntTriangle), perimeter t = 8 ∧
  ∀ (t' : IntTriangle), perimeter t' = 8 → congruent t t' := by
  sorry

end unique_triangle_with_perimeter_8_l1438_143800


namespace collection_for_37_members_l1438_143845

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection for 37 members is 13.69 rupees -/
theorem collection_for_37_members :
  total_collection_rupees 37 100 = 13.69 := by
  sorry

#eval total_collection_rupees 37 100

end collection_for_37_members_l1438_143845


namespace quadratic_statements_l1438_143888

variable (a b c : ℝ)
variable (x₀ : ℝ)

def quadratic_equation (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_statements (h : a ≠ 0) :
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ u v, u ≠ v ∧ quadratic_equation u = 0 ∧ quadratic_equation v = 0) ∧
  (quadratic_equation x₀ = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by
  sorry

end quadratic_statements_l1438_143888


namespace third_group_size_l1438_143867

theorem third_group_size (total : ℕ) (first_fraction : ℚ) (second_fraction : ℚ)
  (h_total : total = 45)
  (h_first : first_fraction = 1 / 3)
  (h_second : second_fraction = 2 / 5)
  : total - (total * first_fraction).floor - (total * second_fraction).floor = 12 :=
by sorry

end third_group_size_l1438_143867


namespace quadratic_roots_expression_l1438_143832

theorem quadratic_roots_expression (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + 5*x₁ + 1 = 0) 
  (h2 : x₂^2 + 5*x₂ + 1 = 0) : 
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
  sorry

end quadratic_roots_expression_l1438_143832


namespace hyperbola_asymptote_tangent_circle_l1438_143898

/-- The value of k for which the asymptotes of the hyperbola x^2 - y^2/k^2 = 1 
    are tangent to the circle x^2 + (y-2)^2 = 1 -/
theorem hyperbola_asymptote_tangent_circle (k : ℝ) :
  k > 0 →
  (∀ x y : ℝ, x^2 - y^2/k^2 = 1 → 
    ∃ m : ℝ, (∀ t : ℝ, (x = t ∧ y = k*t) ∨ (x = t ∧ y = -k*t)) →
      (∃ x₀ y₀ : ℝ, x₀^2 + (y₀-2)^2 = 1 ∧
        (x₀ - x)^2 + (y₀ - y)^2 = 1)) →
  k = Real.sqrt 3 :=
sorry

end hyperbola_asymptote_tangent_circle_l1438_143898


namespace lansing_elementary_schools_l1438_143829

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 6175 / 247

/-- Theorem: There are 25 elementary schools in Lansing -/
theorem lansing_elementary_schools : num_schools = 25 := by
  sorry

end lansing_elementary_schools_l1438_143829


namespace min_sum_of_digits_f_l1438_143830

def f (n : ℕ) : ℕ := 17 * n^2 - 11 * n + 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem min_sum_of_digits_f :
  ∀ n : ℕ, sum_of_digits (f n) ≥ 2 :=
by sorry

end min_sum_of_digits_f_l1438_143830


namespace rectangle_perimeter_l1438_143828

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a * b = 4 * (2 * a + 2 * b) - 12 → 
  2 * (a + b) = 72 ∨ 2 * (a + b) = 100 := by
sorry

end rectangle_perimeter_l1438_143828


namespace parabola_hyperbola_equations_l1438_143874

/-- A parabola and hyperbola with specific properties -/
structure ParabolaHyperbolaPair where
  -- Parabola properties
  parabola_vertex : ℝ × ℝ
  parabola_axis_through_focus : Bool
  parabola_perpendicular : Bool
  
  -- Hyperbola properties
  hyperbola_a : ℝ
  hyperbola_b : ℝ
  
  -- Intersection point
  intersection : ℝ × ℝ
  
  -- Conditions
  vertex_at_origin : parabola_vertex = (0, 0)
  axis_through_focus : parabola_axis_through_focus = true
  perpendicular_to_real_axis : parabola_perpendicular = true
  intersection_point : intersection = (3/2, Real.sqrt 6)
  hyperbola_equation : ∀ x y, x^2 / hyperbola_a^2 - y^2 / hyperbola_b^2 = 1 → 
    (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)

/-- The equations of the parabola and hyperbola given the conditions -/
theorem parabola_hyperbola_equations (ph : ParabolaHyperbolaPair) :
  (∀ x y, y^2 = 4*x ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∀ x y, x^2 / (1/4) - y^2 / (3/4) = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) :=
sorry

end parabola_hyperbola_equations_l1438_143874


namespace reciprocal_of_negative_2023_l1438_143842

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l1438_143842


namespace area_R_specific_rhombus_l1438_143876

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ := sorry

/-- Theorem: The area of region R in a rhombus with side length 3 and angle B = 150° -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 }
  area_R r = (9 * (Real.sqrt 6 - Real.sqrt 2)) / 8 := by sorry

end area_R_specific_rhombus_l1438_143876


namespace least_subtrahend_for_divisibility_problem_solution_l1438_143807

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ (k : Nat), k < 37 ∧ (1234567 - k) % 37 = 0 ∧ ∀ (m : Nat), m < k → (1234567 - m) % 37 ≠ 0 ∧ k = 13 :=
by sorry

end least_subtrahend_for_divisibility_problem_solution_l1438_143807


namespace range_of_a_l1438_143871

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end range_of_a_l1438_143871


namespace binomial_11_10_l1438_143810

theorem binomial_11_10 : Nat.choose 11 10 = 11 := by
  sorry

end binomial_11_10_l1438_143810


namespace circle_equation_l1438_143882

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def satisfiesConditions (c : Circle) : Prop :=
  let (a, b) := c.center
  -- Condition 1: chord on y-axis has length 2
  c.radius^2 = a^2 + 1 ∧
  -- Condition 2: ratio of arc lengths divided by x-axis is 3:1
  (c.radius^2 = 2 * b^2) ∧
  -- Condition 3: distance from center to line x - 2y = 0 is √5/5
  |a - 2*b| / Real.sqrt 5 = Real.sqrt 5 / 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  satisfiesConditions c →
  ((∃ x y, (x + 1)^2 + (y + 1)^2 = 2) ∨ (∃ x y, (x - 1)^2 + (y - 1)^2 = 2)) :=
by sorry

end circle_equation_l1438_143882


namespace correct_rainwater_collection_l1438_143890

/-- Represents the water collection problem --/
structure WaterCollection where
  tankCapacity : ℕ        -- Tank capacity in liters
  riverWater : ℕ          -- Water collected from river daily in milliliters
  daysToFill : ℕ          -- Number of days to fill the tank
  rainWater : ℕ           -- Water collected from rain daily in milliliters

/-- Theorem stating the correct amount of rainwater collected daily --/
theorem correct_rainwater_collection (w : WaterCollection) 
  (h1 : w.tankCapacity = 50)
  (h2 : w.riverWater = 1700)
  (h3 : w.daysToFill = 20)
  : w.rainWater = 800 := by
  sorry

#check correct_rainwater_collection

end correct_rainwater_collection_l1438_143890


namespace square_wood_weight_l1438_143850

/-- Represents the properties of a piece of wood -/
structure Wood where
  length : ℝ
  width : ℝ
  weight : ℝ

/-- Calculates the area of a piece of wood -/
def area (w : Wood) : ℝ := w.length * w.width

/-- Theorem stating the weight of the square piece of wood -/
theorem square_wood_weight (rect : Wood) (square : Wood) :
  rect.length = 4 ∧ 
  rect.width = 6 ∧ 
  rect.weight = 24 ∧
  square.length = 5 ∧
  square.width = 5 →
  square.weight = 25 := by
  sorry

end square_wood_weight_l1438_143850


namespace ali_baba_max_coins_l1438_143809

/-- Represents the game state -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- The initial game state -/
def initialState : GameState :=
  { piles := List.replicate 10 10 }

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : GameState :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) : GameState :=
  sorry

/-- Play the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState :=
  sorry

/-- Calculate the maximum number of coins Ali Baba can take -/
def maxCoinsAliBaba (finalState : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem ali_baba_max_coins :
  ∃ (rounds : Nat), maxCoinsAliBaba (playGame rounds) = 72 :=
sorry

end ali_baba_max_coins_l1438_143809


namespace final_digit_is_two_l1438_143896

/-- Represents the state of the board with counts of 0s, 1s, and 2s -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents a valid operation on the board -/
inductive Operation
  | erase_zero_one_add_two
  | erase_one_two_add_zero
  | erase_zero_two_add_one

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_zero_one_add_two => 
      ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.erase_one_two_add_zero => 
      ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩
  | Operation.erase_zero_two_add_one => 
      ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def has_one_digit (state : BoardState) : Prop :=
  (state.zeros = 1 ∧ state.ones = 0 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 1 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 0 ∧ state.twos = 1)

/-- The main theorem to prove -/
theorem final_digit_is_two 
  (initial : BoardState) 
  (operations : List Operation) 
  (h_final : has_one_digit (operations.foldl apply_operation initial)) :
  (operations.foldl apply_operation initial).twos = 1 :=
sorry

end final_digit_is_two_l1438_143896


namespace cubic_trinomial_degree_l1438_143841

theorem cubic_trinomial_degree (n : ℕ) : 
  (∃ (p : Polynomial ℝ), p = X^n - 5*X + 4 ∧ Polynomial.degree p = 3) → n = 3 :=
by sorry

end cubic_trinomial_degree_l1438_143841


namespace circles_equal_radii_l1438_143859

/-- Proves that the radii of circles A, B, and C are equal -/
theorem circles_equal_radii (r_A : ℝ) (d_B : ℝ) (c_C : ℝ) : 
  r_A = 5 → d_B = 10 → c_C = 10 * Real.pi → r_A = d_B / 2 ∧ r_A = c_C / (2 * Real.pi) := by
  sorry

end circles_equal_radii_l1438_143859


namespace george_eggs_boxes_l1438_143806

/-- Given a total number of eggs and eggs per box, calculates the number of boxes required. -/
def calculate_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) : ℕ :=
  total_eggs / eggs_per_box

theorem george_eggs_boxes :
  let total_eggs : ℕ := 15
  let eggs_per_box : ℕ := 3
  calculate_boxes total_eggs eggs_per_box = 5 := by
  sorry

end george_eggs_boxes_l1438_143806


namespace binomial_10_3_l1438_143848

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l1438_143848


namespace average_production_before_today_l1438_143818

theorem average_production_before_today 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 45) :
  (n * (n + 1) * new_average - (n + 1) * today_production) / n = 40 :=
by sorry

end average_production_before_today_l1438_143818


namespace arithmetic_mean_of_fractions_l1438_143885

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 5/8
  let b : ℚ := 7/8
  let c : ℚ := 3/4
  c = (a + b) / 2 :=
by sorry

end arithmetic_mean_of_fractions_l1438_143885


namespace total_fireworks_count_l1438_143833

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end total_fireworks_count_l1438_143833


namespace box_paint_area_l1438_143831

/-- The total area to paint inside a cuboid box -/
def total_paint_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height) + length * width

/-- Theorem: The total area to paint inside a cuboid box with dimensions 18 cm long, 10 cm wide, and 2 cm high is 292 square centimeters -/
theorem box_paint_area :
  total_paint_area 18 10 2 = 292 := by
  sorry

end box_paint_area_l1438_143831


namespace largest_of_eight_consecutive_integers_l1438_143889

theorem largest_of_eight_consecutive_integers (n : ℕ) : 
  (n > 0) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 3224) →
  (n + 7 = 406) := by
sorry

end largest_of_eight_consecutive_integers_l1438_143889


namespace min_teams_in_tournament_l1438_143870

/-- Represents a football team in the tournament -/
structure Team where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates the score of a team -/
def score (t : Team) : Nat := 3 * t.wins + t.draws

/-- Represents a football tournament -/
structure Tournament where
  teams : List Team
  /-- Each team plays against every other team once -/
  matches_played : ∀ t ∈ teams, t.wins + t.draws + t.losses = teams.length - 1
  /-- There exists a team with the highest score -/
  highest_scorer : ∃ t ∈ teams, ∀ t' ∈ teams, t ≠ t' → score t > score t'
  /-- The highest scoring team has the fewest wins -/
  fewest_wins : ∃ t ∈ teams, (∀ t' ∈ teams, score t ≥ score t') ∧ 
                              (∀ t' ∈ teams, t ≠ t' → t.wins < t'.wins)

/-- The minimum number of teams in a valid tournament is 8 -/
theorem min_teams_in_tournament : 
  ∀ t : Tournament, t.teams.length ≥ 8 ∧ 
  (∃ t' : Tournament, t'.teams.length = 8) := by sorry

end min_teams_in_tournament_l1438_143870


namespace ascending_order_l1438_143879

-- Define the variables
def a : ℕ := 2^55
def b : ℕ := 3^44
def c : ℕ := 5^33
def d : ℕ := 6^22

-- Theorem stating the ascending order
theorem ascending_order : a < d ∧ d < b ∧ b < c := by sorry

end ascending_order_l1438_143879


namespace teresas_age_at_birth_l1438_143802

/-- Proves Teresa's age when Michiko was born, given the current ages and Morio's age at Michiko's birth -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

#check teresas_age_at_birth

end teresas_age_at_birth_l1438_143802


namespace gum_cost_1000_l1438_143856

/-- The cost of buying a given number of pieces of gum, considering bulk discount --/
def gumCost (pieces : ℕ) : ℚ :=
  let baseCost := 2 * pieces
  let discountedCost := if pieces > 500 then baseCost * (9/10) else baseCost
  discountedCost / 100

theorem gum_cost_1000 :
  gumCost 1000 = 18 := by sorry

end gum_cost_1000_l1438_143856


namespace rainbow_preschool_students_l1438_143844

theorem rainbow_preschool_students (half_day_percent : ℝ) (full_day_count : ℕ) : 
  half_day_percent = 0.25 →
  full_day_count = 60 →
  ∃ total_students : ℕ, 
    (1 - half_day_percent) * (total_students : ℝ) = full_day_count ∧
    total_students = 80 :=
by sorry

end rainbow_preschool_students_l1438_143844


namespace photo_calculation_l1438_143836

theorem photo_calculation (total_photos : ℕ) (claire_photos : ℕ) : 
  (claire_photos : ℚ) + 3 * claire_photos + 5/4 * claire_photos + 
  5/2 * 5/4 * claire_photos + (claire_photos + 3 * claire_photos) / 2 + 
  (claire_photos + 3 * claire_photos) / 4 = total_photos ∧ total_photos = 840 → 
  claire_photos = 74 := by
sorry

end photo_calculation_l1438_143836


namespace liam_fourth_week_l1438_143868

/-- A sequence of four numbers representing chapters read each week -/
def ChapterSequence := Fin 4 → ℕ

/-- The properties of Liam's reading sequence -/
def IsLiamSequence (s : ChapterSequence) : Prop :=
  (∀ i : Fin 3, s (i + 1) = s i + 3) ∧
  (s 0 + s 1 + s 2 + s 3 = 50)

/-- Theorem stating that the fourth number in Liam's sequence is 17 -/
theorem liam_fourth_week (s : ChapterSequence) (h : IsLiamSequence s) : s 3 = 17 := by
  sorry

end liam_fourth_week_l1438_143868


namespace student_count_l1438_143849

theorem student_count : ∃ S : ℕ, 
  S = 92 ∧ 
  (3 / 8 : ℚ) * (S - 20 : ℚ) = 27 := by
  sorry

end student_count_l1438_143849


namespace inequality_region_l1438_143873

open Real

theorem inequality_region (x y : ℝ) : 
  (x^5 - 13*x^3 + 36*x) * (x^4 - 17*x^2 + 16) / 
  ((y^5 - 13*y^3 + 36*y) * (y^4 - 17*y^2 + 16)) ≥ 0 ↔ 
  y ≠ 0 ∧ y ≠ 1 ∧ y ≠ -1 ∧ y ≠ 2 ∧ y ≠ -2 ∧ y ≠ 3 ∧ y ≠ -3 ∧ y ≠ 4 ∧ y ≠ -4 :=
by sorry

end inequality_region_l1438_143873


namespace D_72_l1438_143840

/-- D(n) represents the number of ways to write n as a product of integers > 1, where order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) = 48 -/
theorem D_72 : D 72 = 48 := by sorry

end D_72_l1438_143840


namespace tile1_in_position_B_l1438_143857

-- Define a tile with numbers on its sides
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the four tiles
def tile1 : Tile := ⟨5, 3, 2, 4⟩
def tile2 : Tile := ⟨3, 1, 5, 2⟩
def tile3 : Tile := ⟨4, 0, 6, 5⟩
def tile4 : Tile := ⟨2, 4, 3, 0⟩

-- Define the possible positions
inductive Position
  | A | B | C | D

-- Function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile 1 must be in position B
theorem tile1_in_position_B :
  ∃ (p2 p3 p4 : Position), 
    p2 ≠ Position.B ∧ p3 ≠ Position.B ∧ p4 ≠ Position.B ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (canBeAdjacent tile1 tile2 → 
      (p2 = Position.A ∨ p2 = Position.C ∨ p2 = Position.D)) ∧
    (canBeAdjacent tile1 tile3 → 
      (p3 = Position.A ∨ p3 = Position.C ∨ p3 = Position.D)) ∧
    (canBeAdjacent tile1 tile4 → 
      (p4 = Position.A ∨ p4 = Position.C ∨ p4 = Position.D)) :=
by
  sorry


end tile1_in_position_B_l1438_143857


namespace f_properties_l1438_143862

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

theorem f_properties :
  let f := f
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
    (∃ min_val : ℝ, a = 1 → (∀ y > 0, f 1 y ≥ f 1 2) ∧ f 1 2 = -2 * Real.log 2) ∧
    (
      (a ≥ 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (-2 < a ∧ a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂) ∧
                        (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ > f a x₂) ∧
                        (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a = -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
      (a < -2 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂) ∧
                (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ > f a x₂) ∧
                (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))
    ) := by sorry

end f_properties_l1438_143862


namespace relay_race_distance_per_member_l1438_143865

theorem relay_race_distance_per_member 
  (total_distance : ℕ) (team_members : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_members = 5) : 
  total_distance / team_members = 30 := by
  sorry

end relay_race_distance_per_member_l1438_143865


namespace trapezoid_with_equal_angles_l1438_143897

-- Define a trapezoid
structure Trapezoid :=
  (is_quadrilateral : Bool)
  (has_parallel_sides : Bool)
  (has_nonparallel_sides : Bool)

-- Define properties of a trapezoid
def Trapezoid.is_isosceles (t : Trapezoid) : Prop := sorry
def Trapezoid.is_right_angled (t : Trapezoid) : Prop := sorry
def Trapezoid.has_two_equal_angles (t : Trapezoid) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_equal_angles 
  (t : Trapezoid) 
  (h1 : t.is_quadrilateral = true) 
  (h2 : t.has_parallel_sides = true) 
  (h3 : t.has_nonparallel_sides = true) 
  (h4 : t.has_two_equal_angles) : 
  t.is_isosceles ∨ t.is_right_angled := sorry

end trapezoid_with_equal_angles_l1438_143897


namespace gdp_scientific_notation_l1438_143881

-- Define the value of a trillion
def trillion : ℝ := 10^12

-- Define the GDP value in trillions
def gdp_trillions : ℝ := 121

-- Theorem statement
theorem gdp_scientific_notation :
  gdp_trillions * trillion = 1.21 * 10^14 := by
  sorry

end gdp_scientific_notation_l1438_143881


namespace quarter_percent_of_160_l1438_143860

theorem quarter_percent_of_160 : (1 / 4 : ℚ) / 100 * 160 = (0.4 : ℚ) := by sorry

end quarter_percent_of_160_l1438_143860


namespace viju_aju_age_ratio_l1438_143886

/-- Given that Viju's age 5 years ago was 16 and that four years from now, 
    the ratio of ages of Viju to Aju will be 5:2, 
    prove that the present age ratio of Viju to Aju is 7:2. -/
theorem viju_aju_age_ratio :
  ∀ (viju_age aju_age : ℕ),
    viju_age - 5 = 16 →
    (viju_age + 4) * 2 = (aju_age + 4) * 5 →
    ∃ (k : ℕ), k > 0 ∧ viju_age = 7 * k ∧ aju_age = 2 * k :=
by sorry

end viju_aju_age_ratio_l1438_143886


namespace number_of_d_values_l1438_143801

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def value (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + b * 10 + c

def sum_equation (a b c d : ℕ) : Prop :=
  value a b b c + value b c d b = value d a c d

def one_carryover (a b c d : ℕ) : Prop :=
  (a + b) % 10 = d ∧ a + b ≥ 10

theorem number_of_d_values :
  ∃ (s : Finset ℕ),
    (∀ a b c d : ℕ,
      is_digit a → is_digit b → is_digit c → is_digit d →
      distinct a b c d →
      sum_equation a b c d →
      one_carryover a b c d →
      d ∈ s) ∧
    s.card = 5 := by sorry

end number_of_d_values_l1438_143801


namespace line_x_axis_intersection_l1438_143804

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

/-- Theorem: The intersection point satisfies both the line equation and lies on the x-axis -/
theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end line_x_axis_intersection_l1438_143804


namespace coefficient_x7y2_is_20_l1438_143864

/-- The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 -/
def coefficient_x7y2 : ℕ :=
  (Nat.choose 8 2) - (Nat.choose 8 1)

/-- Theorem: The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 is 20 -/
theorem coefficient_x7y2_is_20 : coefficient_x7y2 = 20 := by
  sorry

end coefficient_x7y2_is_20_l1438_143864


namespace line_intercepts_l1438_143872

/-- Given a line with equation 2x - 3y = 6, prove that its x-intercept is 3 and y-intercept is -2 -/
theorem line_intercepts :
  let line : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y = 6
  ∃ (x y : ℝ), (line x 0 ∧ x = 3) ∧ (line 0 y ∧ y = -2) := by
  sorry

end line_intercepts_l1438_143872


namespace min_value_theorem_l1438_143803

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 3) :
  (1/x + 2/y) ≥ 8/3 :=
sorry

end min_value_theorem_l1438_143803


namespace smallest_positive_difference_l1438_143877

/-- Vovochka's method of adding two three-digit numbers -/
def vovochka_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- The correct sum of two three-digit numbers -/
def correct_sum (a b c d e f : Nat) : Nat :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : Nat) : Int :=
  (vovochka_sum a b c d e f) - (correct_sum a b c d e f)

theorem smallest_positive_difference :
  ∀ a b c d e f : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  sum_difference a b c d e f ≠ 0 →
  1800 ≤ |sum_difference a b c d e f| :=
by sorry

end smallest_positive_difference_l1438_143877


namespace complex_subtraction_l1438_143895

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + I) (h₂ : z₂ = 2 - I) :
  z₁ - z₂ = 1 + 2*I := by
  sorry

end complex_subtraction_l1438_143895


namespace age_ratio_in_two_years_l1438_143811

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 25 years older than his son and the son's current age is 23. -/
theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 23 →
  man_age = son_age + 25 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end age_ratio_in_two_years_l1438_143811


namespace quadratic_roots_real_and_equal_l1438_143847

theorem quadratic_roots_real_and_equal 
  (a k : ℝ) 
  (ha : a > 0) 
  (hk : k > 0) 
  (h_discriminant : (6 * Real.sqrt k) ^ 2 - 4 * a * (18 * k) = 0) : 
  ∃ x : ℝ, ∀ y : ℝ, a * y ^ 2 - 6 * y * Real.sqrt k + 18 * k = 0 ↔ y = x :=
sorry

end quadratic_roots_real_and_equal_l1438_143847


namespace problem_solution_l1438_143893

theorem problem_solution (a b : ℤ) (ha : a = 4) (hb : b = -1) : 
  -a^2 - b^2 + a*b = -21 := by
sorry

end problem_solution_l1438_143893


namespace ball_distribution_theorem_l1438_143851

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def yellow_balls : ℕ := 5
def balls_per_box : ℕ := 5
def red_in_box_A : ℕ := 3
def yellow_in_box_A : ℕ := 2
def exchanged_balls : ℕ := 3

def probability_3_red_2_yellow : ℚ := 25 / 63

def mathematical_expectation : ℚ := 12 / 5

theorem ball_distribution_theorem :
  (probability_3_red_2_yellow = 25 / 63) ∧
  (mathematical_expectation = 12 / 5) := by
  sorry

end ball_distribution_theorem_l1438_143851


namespace age_sum_proof_l1438_143817

/-- Given that Ashley's age is 8 and the ratio of Ashley's age to Mary's age is 4:7,
    prove that the sum of their ages is 22. -/
theorem age_sum_proof (ashley_age mary_age : ℕ) : 
  ashley_age = 8 → 
  ashley_age * 7 = mary_age * 4 → 
  ashley_age + mary_age = 22 := by
sorry

end age_sum_proof_l1438_143817


namespace min_additional_bureaus_for_192_and_36_l1438_143824

/-- Given a number of bureaus and offices, calculates the minimum number of additional
    bureaus needed to ensure each office gets an equal number of bureaus. -/
def min_additional_bureaus (total_bureaus : ℕ) (num_offices : ℕ) : ℕ :=
  let bureaus_per_office := (total_bureaus + num_offices - 1) / num_offices
  bureaus_per_office * num_offices - total_bureaus

/-- Proves that for 192 bureaus and 36 offices, the minimum number of additional
    bureaus needed is 24. -/
theorem min_additional_bureaus_for_192_and_36 :
  min_additional_bureaus 192 36 = 24 := by
  sorry

#eval min_additional_bureaus 192 36

end min_additional_bureaus_for_192_and_36_l1438_143824


namespace tangent_line_equation_l1438_143866

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, f 1)
  let m : ℝ := (3 * P.1^2 - 1)  -- Derivative of f at x = 1
  (2 : ℝ) * x - y + 1 = 0 ↔ y - P.2 = m * (x - P.1) := by sorry

end tangent_line_equation_l1438_143866


namespace max_log_sum_l1438_143815

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ Real.log x + Real.log y) →
  Real.log x + Real.log y = 2 := by
sorry

end max_log_sum_l1438_143815


namespace even_digits_in_base8_523_l1438_143827

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-8 representation of 523₁₀ is 1 -/
theorem even_digits_in_base8_523 :
  countEvenDigits (toBase8 523) = 1 := by
  sorry

end even_digits_in_base8_523_l1438_143827


namespace tetris_score_calculation_l1438_143814

/-- Represents the score calculation for a Tetris game with bonus conditions -/
theorem tetris_score_calculation 
  (single_line_score : ℕ)
  (tetris_score_multiplier : ℕ)
  (single_tetris_bonus_multiplier : ℕ)
  (back_to_back_tetris_bonus : ℕ)
  (single_double_triple_bonus : ℕ)
  (singles_count : ℕ)
  (tetrises_count : ℕ)
  (doubles_count : ℕ)
  (triples_count : ℕ)
  (single_tetris_consecutive_count : ℕ)
  (back_to_back_tetris_count : ℕ)
  (single_double_triple_consecutive_count : ℕ)
  (h1 : single_line_score = 1000)
  (h2 : tetris_score_multiplier = 8)
  (h3 : single_tetris_bonus_multiplier = 2)
  (h4 : back_to_back_tetris_bonus = 5000)
  (h5 : single_double_triple_bonus = 3000)
  (h6 : singles_count = 6)
  (h7 : tetrises_count = 4)
  (h8 : doubles_count = 2)
  (h9 : triples_count = 1)
  (h10 : single_tetris_consecutive_count = 1)
  (h11 : back_to_back_tetris_count = 1)
  (h12 : single_double_triple_consecutive_count = 1) :
  singles_count * single_line_score + 
  tetrises_count * (tetris_score_multiplier * single_line_score) +
  single_tetris_consecutive_count * (single_tetris_bonus_multiplier - 1) * (tetris_score_multiplier * single_line_score) +
  back_to_back_tetris_count * back_to_back_tetris_bonus +
  single_double_triple_consecutive_count * single_double_triple_bonus = 54000 := by
  sorry


end tetris_score_calculation_l1438_143814


namespace whales_next_year_l1438_143858

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in whales for next year -/
def predicted_increase : ℕ := 800

/-- The theorem stating the number of whales next year -/
theorem whales_next_year : whales_this_year + predicted_increase = 8800 := by
  sorry

end whales_next_year_l1438_143858


namespace circles_relation_l1438_143843

theorem circles_relation (a b c : ℝ) :
  (∃ x : ℝ, x^2 - 2*a*x + b^2 = c*(b - a) ∧ 
   ∀ y : ℝ, y^2 - 2*a*y + b^2 = c*(b - a) → y = x) →
  (a = b ∨ c = a + b) :=
by sorry

end circles_relation_l1438_143843


namespace feta_cheese_price_per_pound_l1438_143899

/-- Given Teresa's shopping list and total spent, calculate the price per pound of feta cheese --/
theorem feta_cheese_price_per_pound 
  (sandwich_price : ℝ) 
  (sandwich_quantity : ℕ) 
  (salami_price : ℝ) 
  (olive_price_per_pound : ℝ) 
  (olive_quantity : ℝ) 
  (bread_price : ℝ) 
  (feta_quantity : ℝ) 
  (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : sandwich_quantity = 2)
  (h3 : salami_price = 4)
  (h4 : olive_price_per_pound = 10)
  (h5 : olive_quantity = 0.25)
  (h6 : bread_price = 2)
  (h7 : feta_quantity = 0.5)
  (h8 : total_spent = 40) :
  (total_spent - (sandwich_price * sandwich_quantity + salami_price + 3 * salami_price + 
  olive_price_per_pound * olive_quantity + bread_price)) / feta_quantity = 8 := by
sorry

end feta_cheese_price_per_pound_l1438_143899


namespace probability_of_drawing_heart_l1438_143808

-- Define the total number of cards
def total_cards : ℕ := 5

-- Define the number of heart cards
def heart_cards : ℕ := 3

-- Define the number of spade cards
def spade_cards : ℕ := 2

-- Theorem statement
theorem probability_of_drawing_heart :
  (heart_cards : ℚ) / total_cards = 3 / 5 := by sorry

end probability_of_drawing_heart_l1438_143808


namespace initial_roses_count_l1438_143820

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses after adding more flowers -/
def final_roses : ℕ := 12

/-- The number of orchids after adding more flowers -/
def final_orchids : ℕ := 2

/-- The difference between the number of roses and orchids after adding flowers -/
def rose_orchid_difference : ℕ := 10

theorem initial_roses_count : initial_roses = 2 := by
  sorry

end initial_roses_count_l1438_143820


namespace joseph_total_distance_l1438_143855

/-- Joseph's daily running distance in meters -/
def daily_distance : ℕ := 900

/-- Number of days Joseph ran -/
def days_run : ℕ := 3

/-- Total distance Joseph ran -/
def total_distance : ℕ := daily_distance * days_run

/-- Theorem: Joseph's total running distance is 2700 meters -/
theorem joseph_total_distance : total_distance = 2700 := by
  sorry

end joseph_total_distance_l1438_143855


namespace range_of_quadratic_l1438_143812

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The domain of the function -/
def domain : Set ℝ := Set.Ioc 1 4

/-- The range of the function on the given domain -/
def range : Set ℝ := f '' domain

theorem range_of_quadratic : range = Set.Icc (-1) 3 := by sorry

end range_of_quadratic_l1438_143812
