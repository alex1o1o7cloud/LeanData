import Mathlib

namespace NUMINAMATH_CALUDE_other_color_students_l1135_113503

theorem other_color_students (total : ℕ) (blue_percent red_percent green_percent : ℚ) : 
  total = 800 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  (total : ℚ) * (1 - (blue_percent + red_percent + green_percent)) = 136 := by
  sorry

end NUMINAMATH_CALUDE_other_color_students_l1135_113503


namespace NUMINAMATH_CALUDE_balloons_lost_l1135_113568

theorem balloons_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 9 → current = 7 → lost = initial - current → lost = 2 := by sorry

end NUMINAMATH_CALUDE_balloons_lost_l1135_113568


namespace NUMINAMATH_CALUDE_total_letters_written_l1135_113547

/-- The number of letters Nathan can write in one hour -/
def nathan_speed : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_relative_speed : ℕ := 2

/-- The number of hours they write together -/
def total_hours : ℕ := 10

/-- Theorem stating the total number of letters Jacob and Nathan can write together -/
theorem total_letters_written : 
  (nathan_speed + jacob_relative_speed * nathan_speed) * total_hours = 750 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_written_l1135_113547


namespace NUMINAMATH_CALUDE_sum_x_y_value_l1135_113581

theorem sum_x_y_value (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 17)
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 36 / 85 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_value_l1135_113581


namespace NUMINAMATH_CALUDE_concentric_circles_angle_l1135_113544

theorem concentric_circles_angle (r₁ r₂ : ℝ) (α : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  (((360 - α) / 360 * π * r₁^2) + (α / 360 * π * r₂^2) - (α / 360 * π * r₁^2)) = (1/3) * (π * r₂^2) →
  α = 60 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_angle_l1135_113544


namespace NUMINAMATH_CALUDE_smallest_number_with_five_remainders_l1135_113588

theorem smallest_number_with_five_remainders (n : ℕ) : 
  (∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ n ∧
    a % 11 = 3 ∧ b % 11 = 3 ∧ c % 11 = 3 ∧ d % 11 = 3 ∧ e % 11 = 3 ∧
    ∀ (x : ℕ), x ≤ n → x % 11 = 3 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) ↔
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_five_remainders_l1135_113588


namespace NUMINAMATH_CALUDE_james_vegetable_consumption_l1135_113517

/-- Calculates the final weekly vegetable consumption based on initial daily consumption and changes --/
def final_weekly_consumption (initial_daily : ℚ) (kale_addition : ℚ) : ℚ :=
  (initial_daily * 2 * 7) + kale_addition

/-- Proves that James' final weekly vegetable consumption is 10 pounds --/
theorem james_vegetable_consumption :
  let initial_daily := (1/4 : ℚ) + (1/4 : ℚ)
  let kale_addition := (3 : ℚ)
  final_weekly_consumption initial_daily kale_addition = 10 := by
  sorry

#eval final_weekly_consumption ((1/4 : ℚ) + (1/4 : ℚ)) 3

end NUMINAMATH_CALUDE_james_vegetable_consumption_l1135_113517


namespace NUMINAMATH_CALUDE_chang_e_2_orbit_period_l1135_113556

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (α n + 1 / b α n)

theorem chang_e_2_orbit_period (α : ℕ → ℕ) :
  b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_chang_e_2_orbit_period_l1135_113556


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l1135_113594

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 50 * x + 48 = 0) → (x ≥ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l1135_113594


namespace NUMINAMATH_CALUDE_kelly_has_8_students_l1135_113508

/-- Represents the number of students in Kelly's class -/
def num_students : ℕ := sorry

/-- Represents the number of construction paper pieces needed per student -/
def paper_per_student : ℕ := 3

/-- Represents the number of glue bottles Kelly bought -/
def glue_bottles : ℕ := 6

/-- Represents the number of additional construction paper pieces Kelly bought -/
def additional_paper : ℕ := 5

/-- Represents the total number of supplies Kelly has left -/
def total_supplies : ℕ := 20

/-- Theorem stating that Kelly has 8 students given the conditions -/
theorem kelly_has_8_students :
  (((num_students * paper_per_student + glue_bottles) / 2 + additional_paper) = total_supplies) →
  num_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_has_8_students_l1135_113508


namespace NUMINAMATH_CALUDE_kareem_has_largest_result_l1135_113527

def jose_result (x : ℕ) : ℕ := ((x - 1) * 2) + 2

def thuy_result (x : ℕ) : ℕ := ((x * 2) - 1) + 2

def kareem_result (x : ℕ) : ℕ := ((x - 1) + 2) * 2

theorem kareem_has_largest_result :
  let start := 10
  kareem_result start > jose_result start ∧ kareem_result start > thuy_result start :=
by sorry

end NUMINAMATH_CALUDE_kareem_has_largest_result_l1135_113527


namespace NUMINAMATH_CALUDE_sum_of_squares_l1135_113506

theorem sum_of_squares : 2 * 2009^2 + 2 * 2010^2 = 4019^2 + 1^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1135_113506


namespace NUMINAMATH_CALUDE_line_equation_proof_l1135_113530

/-- Given a point (2, 1) and a slope of -2, prove that the equation 2x + y - 5 = 0 represents the line passing through this point with the given slope. -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (2, 1)
  let slope : ℝ := -2
  (2 * x + y - 5 = 0) ↔ (y - point.2 = slope * (x - point.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1135_113530


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1135_113541

theorem inequality_system_solution (a b : ℝ) :
  (∀ x, x + a > 1 ∧ 2 * x - b < 2 ↔ -2 < x ∧ x < 3) →
  (a - b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1135_113541


namespace NUMINAMATH_CALUDE_july_birth_percentage_l1135_113529

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 150) (h2 : july_births = 18) : 
  (july_births : ℚ) / total * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l1135_113529


namespace NUMINAMATH_CALUDE_sine_arithmetic_sequence_l1135_113578

open Real

theorem sine_arithmetic_sequence (a : ℝ) : 
  0 < a ∧ a < 2 * π →
  (∃ r : ℝ, sin a + r = sin (2 * a) ∧ sin (2 * a) + r = sin (3 * a)) ↔ 
  a = π / 2 ∨ a = 3 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_arithmetic_sequence_l1135_113578


namespace NUMINAMATH_CALUDE_robbery_participants_l1135_113505

/-- Represents a gangster --/
inductive Gangster : Type
  | Harry : Gangster
  | James : Gangster
  | Donald : Gangster
  | George : Gangster
  | Charlie : Gangster
  | Tom : Gangster

/-- Represents a statement made by a gangster about who participated in the robbery --/
def Statement : Gangster → Gangster × Gangster
  | Gangster.Harry => (Gangster.Charlie, Gangster.George)
  | Gangster.James => (Gangster.Donald, Gangster.Tom)
  | Gangster.Donald => (Gangster.Tom, Gangster.Charlie)
  | Gangster.George => (Gangster.Harry, Gangster.Charlie)
  | Gangster.Charlie => (Gangster.Donald, Gangster.James)
  | Gangster.Tom => (Gangster.Tom, Gangster.Tom)  -- Placeholder statement for Tom

/-- Checks if a gangster's statement is correct given the actual participants --/
def isCorrectStatement (g : Gangster) (participants : Gangster × Gangster) : Prop :=
  (Statement g).1 = participants.1 ∨ (Statement g).1 = participants.2 ∨
  (Statement g).2 = participants.1 ∨ (Statement g).2 = participants.2

/-- The main theorem stating that Charlie and James participated in the robbery --/
theorem robbery_participants :
  ∃ (participants : Gangster × Gangster),
    participants = (Gangster.Charlie, Gangster.James) ∧
    participants.1 ≠ Gangster.Tom ∧
    participants.2 ≠ Gangster.Tom ∧
    (∃ (incorrect : Gangster),
      incorrect ≠ participants.1 ∧
      incorrect ≠ participants.2 ∧
      ¬isCorrectStatement incorrect participants) ∧
    (∀ (g : Gangster),
      g ≠ participants.1 ∧
      g ≠ participants.2 ∧
      g ≠ Gangster.Tom →
      isCorrectStatement g participants) :=
  sorry


end NUMINAMATH_CALUDE_robbery_participants_l1135_113505


namespace NUMINAMATH_CALUDE_parabola_vertex_locus_l1135_113524

/-- The locus of the vertex of a parabola with specific constraints -/
theorem parabola_vertex_locus (a b s t : ℝ) : 
  (∃ x y, y = a * x^2 + b * x + 1) →  -- parabola equation
  (8 * a^2 + 4 * a * b = b^3) →       -- constraint on a and b
  (s = -b / (2 * a)) →                -- x-coordinate of vertex
  (t = (4 * a - b^2) / (4 * a)) →     -- y-coordinate of vertex
  (s * t = 1) :=                      -- locus equation
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_locus_l1135_113524


namespace NUMINAMATH_CALUDE_land_value_calculation_l1135_113533

/-- Proves that if Blake gives Connie $20,000, and the value of the land Connie buys triples in one year,
    then half of the land's value after one year is $30,000. -/
theorem land_value_calculation (initial_amount : ℕ) (value_multiplier : ℕ) : 
  initial_amount = 20000 → value_multiplier = 3 → 
  (initial_amount * value_multiplier) / 2 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_land_value_calculation_l1135_113533


namespace NUMINAMATH_CALUDE_expand_expression_l1135_113526

theorem expand_expression (x : ℝ) : (11 * x + 5) * 3 * x^3 = 33 * x^4 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1135_113526


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1135_113514

theorem cement_mixture_weight :
  ∀ (W : ℝ),
    (1/3 : ℝ) * W + (1/4 : ℝ) * W + 10 = W →
    W = 24 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1135_113514


namespace NUMINAMATH_CALUDE_max_first_day_volume_l1135_113528

def container_volumes : List Nat := [9, 13, 17, 19, 20, 38]

def is_valid_first_day_selection (selection : List Nat) : Prop :=
  selection.length = 3 ∧ selection.all (λ x => x ∈ container_volumes)

def is_valid_second_day_selection (first_day : List Nat) (second_day : List Nat) : Prop :=
  second_day.length = 2 ∧ 
  second_day.all (λ x => x ∈ container_volumes) ∧
  (∀ x ∈ second_day, x ∉ first_day)

def satisfies_volume_constraint (first_day : List Nat) (second_day : List Nat) : Prop :=
  first_day.sum = 2 * second_day.sum

theorem max_first_day_volume : 
  ∃ (first_day second_day : List Nat),
    is_valid_first_day_selection first_day ∧
    is_valid_second_day_selection first_day second_day ∧
    satisfies_volume_constraint first_day second_day ∧
    first_day.sum = 66 ∧
    (∀ (other_first_day : List Nat),
      is_valid_first_day_selection other_first_day →
      other_first_day.sum ≤ 66) :=
by sorry

end NUMINAMATH_CALUDE_max_first_day_volume_l1135_113528


namespace NUMINAMATH_CALUDE_parallel_transitive_l1135_113572

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (l1 l2 l3 : Line) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l1135_113572


namespace NUMINAMATH_CALUDE_star_value_l1135_113598

theorem star_value (x : ℤ) : 45 - (28 - (37 - (15 - x))) = 59 → x = -154 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1135_113598


namespace NUMINAMATH_CALUDE_symmetry_of_product_l1135_113550

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x

def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, g (-x) = g x

theorem symmetry_of_product (f g : ℝ → ℝ) 
    (hf : IsOddFunction f) (hg : IsEvenFunction g) : 
    IsOddFunction (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_product_l1135_113550


namespace NUMINAMATH_CALUDE_cos_squared_pi_twelfth_plus_one_l1135_113520

theorem cos_squared_pi_twelfth_plus_one :
  2 * (Real.cos (π / 12))^2 + 1 = 2 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_twelfth_plus_one_l1135_113520


namespace NUMINAMATH_CALUDE_final_passenger_count_l1135_113545

def bus_passengers (initial : ℕ) (first_stop : ℕ) (off_other : ℕ) (on_other : ℕ) : ℕ :=
  initial + first_stop - off_other + on_other

theorem final_passenger_count :
  bus_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_final_passenger_count_l1135_113545


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l1135_113564

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (nuts_fraction : ℚ) (berries_fraction : ℚ) (cream_fraction : ℚ) (choc_chips_fraction : ℚ)
  (h_total : total_pies = 48)
  (h_nuts : nuts_fraction = 1/3)
  (h_berries : berries_fraction = 1/2)
  (h_cream : cream_fraction = 3/5)
  (h_choc_chips : choc_chips_fraction = 1/4) :
  ∃ (max_without : ℕ), max_without ≤ total_pies ∧ 
  max_without = total_pies - ⌈cream_fraction * total_pies⌉ ∧
  max_without = 19 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l1135_113564


namespace NUMINAMATH_CALUDE_pizzas_per_person_is_30_l1135_113516

/-- The number of croissants each person eats -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats -/
def cakes_per_person : ℕ := 18

/-- The total number of items consumed by both people -/
def total_items : ℕ := 110

/-- The number of people -/
def num_people : ℕ := 2

theorem pizzas_per_person_is_30 :
  ∃ (pizzas_per_person : ℕ),
    pizzas_per_person = 30 ∧
    num_people * (croissants_per_person + cakes_per_person + pizzas_per_person) = total_items :=
by sorry

end NUMINAMATH_CALUDE_pizzas_per_person_is_30_l1135_113516


namespace NUMINAMATH_CALUDE_open_box_volume_l1135_113513

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 7) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5236 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l1135_113513


namespace NUMINAMATH_CALUDE_unsafe_overtaking_l1135_113567

/-- Represents the safety of car A overtaking car B before meeting car C -/
def is_overtaking_safe (V_A V_B V_C d_AB d_AC : ℝ) : Prop :=
  let rel_vel_AB := V_A - V_B
  let rel_vel_AC := V_A + V_C
  let t_AB := d_AB / rel_vel_AB
  let t_AC := d_AC / rel_vel_AC
  t_AB < t_AC

/-- Theorem stating that car A cannot safely overtake car B before meeting car C
    under the given conditions -/
theorem unsafe_overtaking :
  let V_A : ℝ := 55  -- mph
  let V_B : ℝ := 45  -- mph
  let V_C : ℝ := 50  -- mph (10% less than flat road velocity)
  let d_AB : ℝ := 50 -- ft
  let d_AC : ℝ := 200 -- ft
  ¬(is_overtaking_safe V_A V_B V_C d_AB d_AC) := by
  sorry

#check unsafe_overtaking

end NUMINAMATH_CALUDE_unsafe_overtaking_l1135_113567


namespace NUMINAMATH_CALUDE_correct_ticket_sales_l1135_113511

/-- A structure representing ticket sales for different movie genres --/
structure TicketSales where
  romance : ℕ
  horror : ℕ
  action : ℕ
  comedy : ℕ

/-- Definition of valid ticket sales based on the given conditions --/
def is_valid_ticket_sales (t : TicketSales) : Prop :=
  t.romance = 25 ∧
  t.horror = 3 * t.romance + 18 ∧
  t.action = 2 * t.romance ∧
  5 * t.comedy = 4 * t.horror

/-- Theorem stating the correct number of tickets sold for each genre --/
theorem correct_ticket_sales :
  ∃ (t : TicketSales), is_valid_ticket_sales t ∧
    t.horror = 93 ∧ t.action = 50 ∧ t.comedy = 74 := by
  sorry

end NUMINAMATH_CALUDE_correct_ticket_sales_l1135_113511


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l1135_113555

theorem quadratic_solution_product (x : ℝ) : 
  (49 = -2 * x^2 - 8 * x) → (∃ y : ℝ, (49 = -2 * y^2 - 8 * y) ∧ x * y = 49/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l1135_113555


namespace NUMINAMATH_CALUDE_min_cuts_for_polygons_l1135_113561

/-- Represents the number of sides in the target polygons -/
def target_sides : Nat := 20

/-- Represents the number of target polygons to be created -/
def num_polygons : Nat := 3

/-- Represents the initial number of vertices in the rectangular sheet -/
def initial_vertices : Nat := 4

/-- Represents the maximum increase in vertices per cut -/
def max_vertex_increase : Nat := 4

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_polygons : 
  ∃ (n : Nat), n = 50 ∧ 
  (∀ m : Nat, m < n → 
    (m + 1) * initial_vertices + m * max_vertex_increase < 
    num_polygons * target_sides + 3 * (m + 1 - num_polygons)) ∧
  ((n + 1) * initial_vertices + n * max_vertex_increase ≥ 
    num_polygons * target_sides + 3 * (n + 1 - num_polygons)) := by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_polygons_l1135_113561


namespace NUMINAMATH_CALUDE_unique_triple_exists_l1135_113512

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem unique_triple_exists :
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 90 ∧
    Nat.lcm x z = 720 ∧
    Nat.lcm y z = 1000 ∧
    x < y ∧ y < z ∧
    (is_square x ∨ is_square y ∨ is_square z) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_exists_l1135_113512


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l1135_113560

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (0, 5)

-- Define point E on BC
def E : ℝ × ℝ := (5, 2)

-- G is on the diagonal BD
def G : ℝ × ℝ := sorry

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_AGE :
  triangleArea A G E = 43.25 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l1135_113560


namespace NUMINAMATH_CALUDE_nth_term_from_sum_l1135_113562

/-- Given a sequence {a_n} where S_n = 3n^2 - 2n is the sum of its first n terms,
    prove that the n-th term of the sequence is a_n = 6n - 5 for all natural numbers n. -/
theorem nth_term_from_sum (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ k, S k = 3 * k^2 - 2 * k) : 
  a n = 6 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_nth_term_from_sum_l1135_113562


namespace NUMINAMATH_CALUDE_clara_score_reversal_l1135_113536

theorem clara_score_reversal (a b : ℕ) :
  (∃ (second_game third_game : ℕ),
    second_game = 45 ∧
    third_game = 54 ∧
    (10 * b + a) + second_game + third_game = (10 * a + b) + second_game + third_game + 132) →
  (10 * b + a) - (10 * a + b) = 126 :=
by sorry

end NUMINAMATH_CALUDE_clara_score_reversal_l1135_113536


namespace NUMINAMATH_CALUDE_iggy_monday_run_l1135_113591

/-- Represents the number of miles run on each day of the week --/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total miles run in a week --/
def totalMiles (run : WeeklyRun) : ℝ :=
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday

/-- Represents Iggy's running schedule for the week --/
def iggyRun : WeeklyRun where
  monday := 3  -- This is what we want to prove
  tuesday := 4
  wednesday := 6
  thursday := 8
  friday := 3

/-- Iggy's pace in minutes per mile --/
def pace : ℝ := 10

/-- Total time Iggy spent running in minutes --/
def totalTime : ℝ := 4 * 60

theorem iggy_monday_run :
  iggyRun.monday = 3 ∧ 
  totalMiles iggyRun * pace = totalTime := by
  sorry


end NUMINAMATH_CALUDE_iggy_monday_run_l1135_113591


namespace NUMINAMATH_CALUDE_gelato_sundae_combinations_l1135_113531

theorem gelato_sundae_combinations :
  (Finset.univ.filter (fun s : Finset (Fin 8) => s.card = 3)).card = 56 := by
  sorry

end NUMINAMATH_CALUDE_gelato_sundae_combinations_l1135_113531


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1135_113500

def a : ℝ × ℝ := (2, 1)
def b (t : ℝ) : ℝ × ℝ := (4, t)

theorem parallel_vectors_t_value :
  ∀ t : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a = k • b t) → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1135_113500


namespace NUMINAMATH_CALUDE_johnson_family_seating_theorem_l1135_113539

def johnson_family_seating (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) - (Nat.factorial n * Nat.factorial m)

theorem johnson_family_seating_theorem :
  johnson_family_seating 5 4 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_theorem_l1135_113539


namespace NUMINAMATH_CALUDE_sally_cost_is_42000_l1135_113590

/-- The cost of Lightning McQueen in dollars -/
def lightning_cost : ℝ := 140000

/-- The cost of Mater as a percentage of Lightning McQueen's cost -/
def mater_percentage : ℝ := 0.10

/-- The factor by which Sally McQueen's cost is greater than Mater's cost -/
def sally_factor : ℝ := 3

/-- The cost of Sally McQueen in dollars -/
def sally_cost : ℝ := lightning_cost * mater_percentage * sally_factor

theorem sally_cost_is_42000 : sally_cost = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_cost_is_42000_l1135_113590


namespace NUMINAMATH_CALUDE_evaluate_expression_l1135_113584

theorem evaluate_expression : (2^3002 * 3^3004) / 6^3003 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1135_113584


namespace NUMINAMATH_CALUDE_mount_pilot_snow_amount_l1135_113518

/-- The amount of snow on Mount Pilot in centimeters -/
def mount_pilot_snow (bald_snow billy_snow : ℝ) : ℝ :=
  (billy_snow * 100 + (billy_snow * 100 + bald_snow * 100 + 326) - bald_snow * 100) - billy_snow * 100

/-- Theorem stating that Mount Pilot received 326 cm of snow -/
theorem mount_pilot_snow_amount :
  mount_pilot_snow 1.5 3.5 = 326 := by
  sorry

#eval mount_pilot_snow 1.5 3.5

end NUMINAMATH_CALUDE_mount_pilot_snow_amount_l1135_113518


namespace NUMINAMATH_CALUDE_kathleen_july_savings_l1135_113553

/-- Represents Kathleen's savings and expenses --/
structure KathleenFinances where
  june_savings : ℕ
  august_savings : ℕ
  school_supplies_expense : ℕ
  clothes_expense : ℕ
  money_left : ℕ
  aunt_bonus_threshold : ℕ

/-- Calculates Kathleen's savings in July --/
def july_savings (k : KathleenFinances) : ℕ :=
  k.school_supplies_expense + k.clothes_expense + k.money_left - k.june_savings - k.august_savings

/-- Theorem stating that Kathleen's savings in July is $46 --/
theorem kathleen_july_savings :
  ∀ (k : KathleenFinances),
    k.june_savings = 21 →
    k.august_savings = 45 →
    k.school_supplies_expense = 12 →
    k.clothes_expense = 54 →
    k.money_left = 46 →
    k.aunt_bonus_threshold = 125 →
    july_savings k = 46 :=
  sorry


end NUMINAMATH_CALUDE_kathleen_july_savings_l1135_113553


namespace NUMINAMATH_CALUDE_largest_x_value_l1135_113576

theorem largest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → x ≤ 1) ∧
  (∃ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l1135_113576


namespace NUMINAMATH_CALUDE_circle_min_radius_l1135_113519

theorem circle_min_radius (a : ℝ) : 
  let r := Real.sqrt ((5 * a^2) / 4 + 2)
  r ≥ Real.sqrt 2 ∧ ∃ a₀, Real.sqrt ((5 * a₀^2) / 4 + 2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_min_radius_l1135_113519


namespace NUMINAMATH_CALUDE_park_group_problem_l1135_113537

theorem park_group_problem (girls boys : ℕ) (groups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  groups = 3 → 
  group_size = 25 → 
  groups * group_size = girls + boys + (groups * group_size - (girls + boys)) →
  groups * group_size - (girls + boys) = 50 := by
  sorry

#check park_group_problem

end NUMINAMATH_CALUDE_park_group_problem_l1135_113537


namespace NUMINAMATH_CALUDE_concyclic_projections_l1135_113501

-- Define a circle and a point on a plane
variable (Circle : Type) (Point : Type)

-- Define a function to check if points are concyclic
variable (are_concyclic : Circle → List Point → Prop)

-- Define a function for orthogonal projection
variable (orthogonal_projection : Point → Point → Point → Point)

-- Theorem statement
theorem concyclic_projections
  (A B C D A' B' C' D' : Point) (circle : Circle) :
  are_concyclic circle [A, B, C, D] →
  A' = orthogonal_projection A B D →
  C' = orthogonal_projection C B D →
  B' = orthogonal_projection B A C →
  D' = orthogonal_projection D A C →
  ∃ (circle' : Circle), are_concyclic circle' [A', B', C', D'] :=
by sorry

end NUMINAMATH_CALUDE_concyclic_projections_l1135_113501


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1135_113571

/-- The common ratio of the infinite geometric series (-4/7) + (14/3) + (-98/9) + ... -/
def common_ratio : ℚ := -49/6

/-- The first term of the geometric series -/
def a₁ : ℚ := -4/7

/-- The second term of the geometric series -/
def a₂ : ℚ := 14/3

/-- The third term of the geometric series -/
def a₃ : ℚ := -98/9

theorem geometric_series_common_ratio :
  (a₂ / a₁ = common_ratio) ∧ (a₃ / a₂ = common_ratio) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1135_113571


namespace NUMINAMATH_CALUDE_rectangle_area_l1135_113507

theorem rectangle_area (AB CD : ℝ) (h1 : AB = 15) (h2 : AB^2 + CD^2 = 17^2) :
  AB * CD = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1135_113507


namespace NUMINAMATH_CALUDE_digit_700_of_3_11_is_7_l1135_113504

/-- The 700th digit past the decimal point in the decimal expansion of 3/11 -/
def digit_700_of_3_11 : ℕ :=
  -- Define the digit here
  7

/-- Theorem stating that the 700th digit past the decimal point
    in the decimal expansion of 3/11 is 7 -/
theorem digit_700_of_3_11_is_7 :
  digit_700_of_3_11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_700_of_3_11_is_7_l1135_113504


namespace NUMINAMATH_CALUDE_train_passing_platform_l1135_113546

/-- The time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  platform_length = 390 →
  train_speed_kmh = 45 →
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l1135_113546


namespace NUMINAMATH_CALUDE_intersection_x_product_l1135_113558

/-- Given a line y = mx + k and a parabola y = ax² + bx + c that intersect at two points,
    the product of the x-coordinates of these intersection points is equal to (c - k) / a. -/
theorem intersection_x_product (a m b c k : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := m * x + k
  let h (x : ℝ) := f x - g x
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 →
  x₁ * x₂ = (c - k) / a :=
sorry

end NUMINAMATH_CALUDE_intersection_x_product_l1135_113558


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1135_113586

/-- Given a positive geometric sequence {a_n} where a_7 = a_6 + 2a_5, 
    and there exist two terms a_m and a_n such that √(a_m * a_n) = 4a_1,
    the minimum value of 1/m + 4/n is 3/2. -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∃ q > 0, ∀ k, a (k + 1) = q * a k) →  -- Geometric sequence
  a 7 = a 6 + 2 * a 5 →  -- Given condition
  Real.sqrt (a m * a n) = 4 * a 1 →  -- Given condition
  (∀ i j : ℕ, 1 / i + 4 / j ≥ 3 / 2) ∧  -- Minimum value is at least 3/2
  (∃ i j : ℕ, 1 / i + 4 / j = 3 / 2) :=  -- Minimum value of 3/2 is achievable
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1135_113586


namespace NUMINAMATH_CALUDE_total_amount_distributed_l1135_113575

/-- Represents the share distribution among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The given share distribution -/
def given_distribution : ShareDistribution :=
  { w := 2
    x := 1.5
    y := 2.5
    z := 1.7 }

/-- Y's share in rupees -/
def y_share : ℝ := 48.50

/-- Theorem stating the total amount distributed -/
theorem total_amount_distributed : 
  let d := given_distribution
  let unit_value := y_share / d.y
  let total_units := d.w + d.x + d.y + d.z
  total_units * unit_value = 188.08 := by
  sorry

#check total_amount_distributed

end NUMINAMATH_CALUDE_total_amount_distributed_l1135_113575


namespace NUMINAMATH_CALUDE_triangle_area_l1135_113573

/-- Theorem: Area of a triangle with sides 8, 15, and 17 --/
theorem triangle_area (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  (1/2) * a * b = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1135_113573


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l1135_113551

theorem smaller_cubes_count (larger_volume : ℝ) (smaller_volume : ℝ) (surface_area_diff : ℝ) :
  larger_volume = 64 →
  smaller_volume = 1 →
  surface_area_diff = 288 →
  (Real.sqrt (larger_volume ^ (1 / 3 : ℝ)))^2 * 6 +
    surface_area_diff =
    (Real.sqrt (smaller_volume ^ (1 / 3 : ℝ)))^2 * 6 *
    (larger_volume / smaller_volume) :=
by sorry

end NUMINAMATH_CALUDE_smaller_cubes_count_l1135_113551


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l1135_113587

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 90°,
    the ratio of the lateral surface area to the base area is 4. -/
theorem cone_surface_area_ratio (r : ℝ) (h : r > 0) : 
  let R := 4 * r
  let base_area := π * r^2
  let lateral_area := (1/4) * π * R^2
  lateral_area / base_area = 4 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l1135_113587


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l1135_113548

theorem gel_pen_price_ratio (x y : ℕ) (b g : ℝ) :
  x > 0 ∧ y > 0 ∧ b > 0 ∧ g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b :=
by
  sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l1135_113548


namespace NUMINAMATH_CALUDE_ratio_of_averages_l1135_113538

theorem ratio_of_averages (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (4 + 20 + x) / 3 = (y + 16) / 2) :
  x / y = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_averages_l1135_113538


namespace NUMINAMATH_CALUDE_circle_condition_exclusive_shape_condition_l1135_113534

/-- Represents a circle equation -/
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 4*x + a^2 = 0

/-- Represents an ellipse equation -/
def is_ellipse (a : ℝ) : Prop :=
  a > 0 ∧ ∃ (x y : ℝ), (y^2 / 3) + (x^2 / a) = 1

/-- The ellipse has its focus on the y-axis -/
def focus_on_y_axis (a : ℝ) : Prop :=
  is_ellipse a → a < 3

theorem circle_condition (a : ℝ) :
  is_circle a ↔ -2 < a ∧ a < 2 :=
sorry

theorem exclusive_shape_condition (a : ℝ) :
  (is_circle a ∨ is_ellipse a) ∧ ¬(is_circle a ∧ is_ellipse a) ↔
  ((-2 < a ∧ a ≤ 0) ∨ (2 ≤ a ∧ a < 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_exclusive_shape_condition_l1135_113534


namespace NUMINAMATH_CALUDE_quadratic_polynomial_form_l1135_113521

/-- A quadratic polynomial with specific properties -/
structure QuadraticPolynomial where
  p : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  value_at_neg_two : p (-2) = 8
  asymptotes : Set ℝ
  asymptotes_def : asymptotes = {-2, 2}
  is_asymptote : ∀ x ∈ asymptotes, ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |1 / (p y)| > 1 / ε

/-- The theorem stating the specific form of the quadratic polynomial -/
theorem quadratic_polynomial_form (f : QuadraticPolynomial) : f.p = λ x => -2 * x^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_form_l1135_113521


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l1135_113597

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 5 * a + 3 * b ≤ 11) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 23 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l1135_113597


namespace NUMINAMATH_CALUDE_relationship_abc_l1135_113542

theorem relationship_abc (a b c : ℝ) : 
  a = (0.4 : ℝ) ^ (0.2 : ℝ) → 
  b = (0.4 : ℝ) ^ (0.6 : ℝ) → 
  c = (2.1 : ℝ) ^ (0.2 : ℝ) → 
  c > a ∧ a > b := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1135_113542


namespace NUMINAMATH_CALUDE_sara_quarters_l1135_113566

theorem sara_quarters (initial : ℕ) : 
  initial + 49 = 70 → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l1135_113566


namespace NUMINAMATH_CALUDE_min_jellybeans_jellybeans_solution_l1135_113595

theorem min_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 :=
by
  sorry

theorem jellybeans_solution : ∃ (n : ℕ), n = 164 ∧ n ≥ 150 ∧ n % 15 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_min_jellybeans_jellybeans_solution_l1135_113595


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1135_113569

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value of 254 in base 8 --/
def num1 : Nat := to_base_10 [4, 5, 2] 8

/-- The value of 16 in base 4 --/
def num2 : Nat := to_base_10 [6, 1] 4

/-- The value of 232 in base 7 --/
def num3 : Nat := to_base_10 [2, 3, 2] 7

/-- The value of 34 in base 5 --/
def num4 : Nat := to_base_10 [4, 3] 5

/-- The main theorem to prove --/
theorem base_conversion_sum :
  (num1 : ℚ) / num2 + (num3 : ℚ) / num4 = 23.6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1135_113569


namespace NUMINAMATH_CALUDE_evenAdjacentCellsCount_l1135_113585

/-- The number of cells with an even number of adjacent cells in an equilateral triangle -/
def evenAdjacentCells (sideLength : ℕ) : ℕ :=
  sideLength * sideLength - (sideLength - 3) * (sideLength - 3) - 3

/-- The side length of the large equilateral triangle -/
def largeSideLength : ℕ := 2022

theorem evenAdjacentCellsCount :
  evenAdjacentCells largeSideLength = 12120 := by
  sorry

end NUMINAMATH_CALUDE_evenAdjacentCellsCount_l1135_113585


namespace NUMINAMATH_CALUDE_hours_to_seconds_l1135_113554

-- Define constants
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the theorem
theorem hours_to_seconds (hours : ℚ) : 
  hours * (minutes_per_hour * seconds_per_minute) = 12600 ↔ hours = 3.5 :=
by sorry

end NUMINAMATH_CALUDE_hours_to_seconds_l1135_113554


namespace NUMINAMATH_CALUDE_periodic_function_value_l1135_113559

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2014 = 5 → f 2015 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1135_113559


namespace NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l1135_113557

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 7 * x - 8 * y - 1 = 0
def l2 (x y : ℝ) : Prop := 2 * x + 17 * y + 9 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 27 * x + 54 * y + 37 = 0

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    (l1 x y ∧ l2 x y) ∧  -- Intersection point satisfies both l1 and l2
    (∀ (x' y' : ℝ), result_line x' y' ↔ 
      (y' - y) = -(1/2) * (x' - x)) ∧  -- Slope of result_line is -1/2
    (∀ (x' y' : ℝ), perp_line x' y' → 
      (y' - y) * (1/2) + (x' - x) = 0)  -- Perpendicular to perp_line
    := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l1135_113557


namespace NUMINAMATH_CALUDE_no_three_reals_satisfying_conditions_l1135_113580

/-- Definition of the set S(a) -/
def S (a : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * a⌋}

/-- Theorem stating the impossibility of finding three positive reals satisfying the given conditions -/
theorem no_three_reals_satisfying_conditions :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧
    (S a ∪ S b ∪ S c = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_no_three_reals_satisfying_conditions_l1135_113580


namespace NUMINAMATH_CALUDE_factorization_equality_l1135_113579

theorem factorization_equality (x : ℝ) : -4 * x^2 + 16 = 4 * (2 + x) * (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1135_113579


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l1135_113583

theorem cookie_boxes_problem (n : ℕ) : 
  (n - 7 ≥ 1) →  -- Mark sold at least one box
  (n - 2 ≥ 1) →  -- Ann sold at least one box
  (n - 3 ≥ 1) →  -- Carol sold at least one box
  ((n - 7) + (n - 2) + (n - 3) < n) →  -- Together they sold less than n boxes
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l1135_113583


namespace NUMINAMATH_CALUDE_saturday_balls_count_l1135_113532

/-- The number of golf balls Corey wants to find every weekend -/
def weekend_goal : ℕ := 48

/-- The number of golf balls Corey found on Sunday -/
def sunday_balls : ℕ := 18

/-- The number of additional golf balls Corey needs to reach his goal -/
def additional_balls_needed : ℕ := 14

/-- The number of golf balls Corey found on Saturday -/
def saturday_balls : ℕ := weekend_goal - sunday_balls - additional_balls_needed

theorem saturday_balls_count : saturday_balls = 16 := by
  sorry

end NUMINAMATH_CALUDE_saturday_balls_count_l1135_113532


namespace NUMINAMATH_CALUDE_bus_back_seat_capacity_l1135_113574

/-- Represents the seating capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  people_per_seat : ℕ
  total_capacity : ℕ

/-- Calculates the number of people who can sit at the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : ℕ :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people who can sit at the back seat of the given bus -/
theorem bus_back_seat_capacity :
  ∃ (bus : BusSeating),
    bus.left_seats = 15 ∧
    bus.right_seats = bus.left_seats - 3 ∧
    bus.people_per_seat = 3 ∧
    bus.total_capacity = 91 ∧
    back_seat_capacity bus = 10 := by
  sorry

end NUMINAMATH_CALUDE_bus_back_seat_capacity_l1135_113574


namespace NUMINAMATH_CALUDE_sector_central_angle_l1135_113596

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1135_113596


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1135_113540

open Set

theorem inequality_solution_set (x : ℝ) :
  let S := {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1 ∧ x ≠ -2}
  S = Ioo (-2 : ℝ) (-1) ∪ Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1135_113540


namespace NUMINAMATH_CALUDE_stating_unique_dissection_l1135_113565

/-- A type representing a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional structure details would go here, but are omitted for simplicity

/-- A type representing a right triangle with integer-ratio sides -/
structure IntegerRatioRightTriangle where
  -- Additional structure details would go here, but are omitted for simplicity

/-- 
Predicate indicating whether a regular n-sided polygon can be 
completely dissected into integer-ratio right triangles 
-/
def canBeDissected (n : ℕ) : Prop :=
  ∃ (p : RegularPolygon n) (triangles : Set IntegerRatioRightTriangle), 
    -- The formal definition of complete dissection would go here
    True  -- Placeholder

/-- 
Theorem stating that 4 is the only integer n ≥ 3 for which 
a regular n-sided polygon can be completely dissected into 
integer-ratio right triangles 
-/
theorem unique_dissection : 
  ∀ n : ℕ, n ≥ 3 → (canBeDissected n ↔ n = 4) := by
  sorry


end NUMINAMATH_CALUDE_stating_unique_dissection_l1135_113565


namespace NUMINAMATH_CALUDE_expression_value_l1135_113599

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 5 * y) / (x - 2 * y) = 26 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1135_113599


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_l1135_113525

theorem cos_difference_from_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = -1)
  (h2 : Real.cos A + Real.cos B = 1/2) : 
  Real.cos (A - B) = -3/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_l1135_113525


namespace NUMINAMATH_CALUDE_gcd_of_324_243_135_l1135_113570

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_135_l1135_113570


namespace NUMINAMATH_CALUDE_greatest_n_roots_on_unit_circle_l1135_113563

theorem greatest_n_roots_on_unit_circle : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z ≠ 0 → (z + 1)^n = z^n + 1 → Complex.abs z = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (w : ℂ), w ≠ 0 ∧ (w + 1)^m = w^m + 1 ∧ Complex.abs w ≠ 1) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_greatest_n_roots_on_unit_circle_l1135_113563


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1135_113522

theorem unique_square_divisible_by_three_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n * n) ∧ 
    (∃ k : ℕ, x = 3 * k) ∧ 
    90 < x ∧ x < 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1135_113522


namespace NUMINAMATH_CALUDE_tech_club_enrollment_l1135_113509

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (robotics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : robotics = 70)
  (h4 : both = 20) :
  total - (cs + robotics - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_enrollment_l1135_113509


namespace NUMINAMATH_CALUDE_A_divisible_by_1980_l1135_113535

def A : ℕ := sorry

theorem A_divisible_by_1980 : 1980 ∣ A := by sorry

end NUMINAMATH_CALUDE_A_divisible_by_1980_l1135_113535


namespace NUMINAMATH_CALUDE_total_teachers_is_182_l1135_113510

/-- Represents the number of teachers in different categories and survey selections -/
structure SchoolTeachers where
  senior : ℕ
  intermediate : ℕ
  survey_total : ℕ
  survey_other : ℕ

/-- Calculates the total number of teachers in the school -/
def total_teachers (s : SchoolTeachers) : ℕ :=
  s.senior + s.intermediate + (s.survey_total - (s.survey_other + s.senior + s.intermediate))

/-- Theorem stating that given the specific numbers, the total teachers is 182 -/
theorem total_teachers_is_182 (s : SchoolTeachers) 
  (h1 : s.senior = 26)
  (h2 : s.intermediate = 104)
  (h3 : s.survey_total = 56)
  (h4 : s.survey_other = 16) :
  total_teachers s = 182 := by
  sorry

#eval total_teachers ⟨26, 104, 56, 16⟩

end NUMINAMATH_CALUDE_total_teachers_is_182_l1135_113510


namespace NUMINAMATH_CALUDE_min_value_of_f_l1135_113543

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ f y a) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ f y a) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1135_113543


namespace NUMINAMATH_CALUDE_circle_center_is_one_one_l1135_113515

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the 2D plane --/
def Point := ℝ × ℝ

/-- The parabola y = x^2 --/
def parabola (x : ℝ) : ℝ := x^2

/-- Check if a point is on a circle --/
def isOnCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the parabola at a given point --/
def isTangentToParabola (c : Circle) (p : Point) : Prop :=
  isOnCircle c p ∧ p.2 = parabola p.1 ∧
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - p.1| ∧ |x - p.1| < ε →
    (x, parabola x) ∉ {q : Point | isOnCircle c q}

theorem circle_center_is_one_one :
  ∀ (c : Circle),
    isOnCircle c (0, 1) →
    isTangentToParabola c (1, 1) →
    c.center = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_one_one_l1135_113515


namespace NUMINAMATH_CALUDE_book_cart_total_l1135_113549

/-- Represents the number of books in each category on the top section of the cart -/
structure TopSection where
  history : ℕ
  romance : ℕ
  poetry : ℕ

/-- Represents the number of books in each category on the bottom section of the cart -/
structure BottomSection where
  western : ℕ
  biography : ℕ
  scifi : ℕ

/-- Represents the entire book cart -/
structure BookCart where
  top : TopSection
  bottom : BottomSection
  mystery : ℕ

def total_books (cart : BookCart) : ℕ :=
  cart.top.history + cart.top.romance + cart.top.poetry +
  cart.bottom.western + cart.bottom.biography + cart.bottom.scifi +
  cart.mystery

theorem book_cart_total (cart : BookCart) :
  cart.top.history = 12 →
  cart.top.romance = 8 →
  cart.top.poetry = 4 →
  cart.bottom.western = 5 →
  cart.bottom.biography = 6 →
  cart.bottom.scifi = 3 →
  cart.mystery = 2 * (cart.bottom.western + cart.bottom.biography + cart.bottom.scifi) →
  total_books cart = 66 := by
  sorry

#check book_cart_total

end NUMINAMATH_CALUDE_book_cart_total_l1135_113549


namespace NUMINAMATH_CALUDE_solution_values_l1135_113593

def A : Set ℝ := {-1, 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 - 2*a*x + b = 0}

theorem solution_values (a b : ℝ) : 
  B a b ≠ ∅ → A ∪ B a b = A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_values_l1135_113593


namespace NUMINAMATH_CALUDE_ellipse_properties_l1135_113552

/-- Given an ellipse C with specific properties, we prove its equation,
    the range of dot product of vectors OA and OB, and a fixed intersection point. -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : Set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let eccentricity : ℝ := Real.sqrt (1 - b^2/a^2)
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = b^2}
  let tangent_line : Set (ℝ × ℝ) := {p | p.1 - p.2 + Real.sqrt 6 = 0}
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/4 + y^2/3 = 1) ∧
    (eccentricity = 1/2) ∧
    (∃ (p : ℝ × ℝ), p ∈ circle ∩ tangent_line) ∧
    (A ∈ C ∧ B ∈ C) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ A.2 - 0 = k * (A.1 - 4) ∧ B.2 - 0 = k * (B.1 - 4)) ∧
    (-4 ≤ (A.1 * B.1 + A.2 * B.2) ∧ (A.1 * B.1 + A.2 * B.2) < 13/4) ∧
    (∃ (E : ℝ × ℝ), E.1 = B.1 ∧ E.2 = -B.2 ∧
      ∃ (t : ℝ), t * A.1 + (1 - t) * E.1 = 1 ∧ t * A.2 + (1 - t) * E.2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1135_113552


namespace NUMINAMATH_CALUDE_lenyas_number_l1135_113582

theorem lenyas_number (x : ℝ) : ((((x + 5) / 3) * 4) - 6) / 7 = 2 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_lenyas_number_l1135_113582


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_half_l1135_113577

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

theorem f_derivative_at_pi_half : 
  deriv f (Real.pi / 2) = -Real.exp (Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_half_l1135_113577


namespace NUMINAMATH_CALUDE_bennys_work_hours_l1135_113589

/-- 
Given that Benny worked 7 hours per day for 14 days, 
prove that his total work hours equals 98.
-/
theorem bennys_work_hours : 
  let hours_per_day : ℕ := 7
  let days_worked : ℕ := 14
  hours_per_day * days_worked = 98 := by sorry

end NUMINAMATH_CALUDE_bennys_work_hours_l1135_113589


namespace NUMINAMATH_CALUDE_range_of_x_l1135_113592

-- Define the function f
def f (x a : ℝ) := |x - 4| + |x - a|

-- State the theorem
theorem range_of_x (a : ℝ) (h1 : a > 1) 
  (h2 : ∃ (m : ℝ), ∀ x, f x a ≥ m ∧ ∃ y, f y a = m) 
  (h3 : (Classical.choose h2) = 3) :
  ∀ x, f x a ≤ 5 → 3 ≤ x ∧ x ≤ 8 := by
  sorry

#check range_of_x

end NUMINAMATH_CALUDE_range_of_x_l1135_113592


namespace NUMINAMATH_CALUDE_n_fourth_plus_four_prime_iff_n_eq_one_l1135_113523

theorem n_fourth_plus_four_prime_iff_n_eq_one (n : ℕ+) :
  Nat.Prime (n^4 + 4) ↔ n = 1 := by
sorry

end NUMINAMATH_CALUDE_n_fourth_plus_four_prime_iff_n_eq_one_l1135_113523


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l1135_113502

/-- The maximum distance between a point on circle1 and a point on circle2 -/
theorem max_distance_between_circles (M N : ℝ × ℝ) : 
  (∃ x y, M = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
  (∃ x y, N = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
  (∀ M' N', 
    (∃ x y, M' = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
    (∃ x y, N' = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
    Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)) →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = (15 + Real.sqrt 23) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l1135_113502
