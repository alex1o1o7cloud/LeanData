import Mathlib

namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l1534_153492

theorem subtraction_of_like_terms (a : ℝ) : 3 * a - a = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l1534_153492


namespace NUMINAMATH_CALUDE_bricklayer_team_size_l1534_153479

theorem bricklayer_team_size :
  ∀ (x : ℕ),
  (x > 4) →
  (432 / x + 9) * (x - 4) = 432 →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bricklayer_team_size_l1534_153479


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l1534_153423

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (hw : walnuts = 0.25)
  (ha : almonds = 0.25) :
  walnuts + almonds = 0.50 := by sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l1534_153423


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l1534_153411

theorem binomial_square_coefficient (x : ℝ) : ∃ (a : ℝ), 
  (∃ (r s : ℝ), (r * x + s)^2 = a * x^2 + 20 * x + 9) ∧ 
  a = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l1534_153411


namespace NUMINAMATH_CALUDE_car_distance_proof_l1534_153451

theorem car_distance_proof (west_speed : ℝ) (east_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  west_speed = 20 →
  east_speed = 60 →
  time = 5 →
  final_distance = 500 →
  ∃ (initial_north_south_distance : ℝ),
    initial_north_south_distance = 300 ∧
    initial_north_south_distance ^ 2 + (west_speed * time + east_speed * time) ^ 2 = final_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1534_153451


namespace NUMINAMATH_CALUDE_laptop_cost_laptop_cost_proof_l1534_153440

/-- The cost of a laptop given the following conditions:
  1. The cost of a smartphone is $400.
  2. Celine buys 2 laptops and 4 smartphones.
  3. Celine pays $3000 and receives $200 in change. -/
theorem laptop_cost : ℕ → Prop :=
  fun laptop_price =>
    let smartphone_price : ℕ := 400
    let laptops_bought : ℕ := 2
    let smartphones_bought : ℕ := 4
    let total_paid : ℕ := 3000
    let change_received : ℕ := 200
    let total_spent : ℕ := total_paid - change_received
    laptop_price * laptops_bought + smartphone_price * smartphones_bought = total_spent ∧
    laptop_price = 600

/-- Proof of the laptop cost theorem -/
theorem laptop_cost_proof : ∃ (x : ℕ), laptop_cost x :=
  sorry

end NUMINAMATH_CALUDE_laptop_cost_laptop_cost_proof_l1534_153440


namespace NUMINAMATH_CALUDE_f_max_value_l1534_153449

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.sin x * Real.cos x + Real.cos x ^ 4

theorem f_max_value : ∀ x : ℝ, f x ≤ 9/8 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1534_153449


namespace NUMINAMATH_CALUDE_min_elements_for_sum_equality_l1534_153499

theorem min_elements_for_sum_equality (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m = 2 * n + 2 ∧
  (∀ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) → S.card ≥ m →
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a = b + c + d) ∧
  (∀ m' : ℕ, m' < m →
    ∃ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) ∧ S.card = m' ∧
    ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a ≠ b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_min_elements_for_sum_equality_l1534_153499


namespace NUMINAMATH_CALUDE_rectangle_border_problem_l1534_153496

theorem rectangle_border_problem : 
  (∃ (S : Finset (ℕ × ℕ)), 
    S.card = 4 ∧ 
    (∀ (a b : ℕ), (a, b) ∈ S ↔ 
      (b > a ∧ 
       a > 0 ∧ 
       b > 0 ∧ 
       (a - 4) * (b - 4) = 2 * (a * b) / 3))) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_border_problem_l1534_153496


namespace NUMINAMATH_CALUDE_pentagon_area_error_percentage_l1534_153461

def actual_side_A : ℝ := 10
def actual_side_B : ℝ := 20
def error_A : ℝ := 0.02
def error_B : ℝ := 0.03

def erroneous_side_A : ℝ := actual_side_A * (1 + error_A)
def erroneous_side_B : ℝ := actual_side_B * (1 - error_B)

def actual_area_factor : ℝ := actual_side_A * actual_side_B
def erroneous_area_factor : ℝ := erroneous_side_A * erroneous_side_B

theorem pentagon_area_error_percentage :
  (erroneous_area_factor - actual_area_factor) / actual_area_factor * 100 = -1.06 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_error_percentage_l1534_153461


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1534_153498

/-- Given an ellipse and a hyperbola with coinciding foci, prove that the parameter d of the ellipse satisfies d² = 667/36 -/
theorem ellipse_hyperbola_coinciding_foci :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / d^2 = 1
  let hyperbola := fun (x y : ℝ) => x^2 / 169 - y^2 / 64 = 1 / 36
  ∀ d : ℝ, (∃ c : ℝ, c^2 = 25 - d^2 ∧ c^2 = 169 / 36 + 64 / 36) →
    d^2 = 667 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1534_153498


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_l1534_153446

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the right focus of the ellipse
def right_focus_ellipse (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the focus of the parabola
def focus_parabola (p x y : ℝ) : Prop := x = p / 2 ∧ y = 0

-- Theorem statement
theorem parabola_ellipse_focus (p : ℝ) :
  (∃ x y : ℝ, right_focus_ellipse x y ∧ focus_parabola p x y) → p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_l1534_153446


namespace NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l1534_153462

/-- The probability of success on a single trial -/
def p : ℚ := 2/3

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coeff n k * p^k * (1-p)^(n-k)

theorem exactly_two_successes_in_four_trials : 
  binomial_prob n k p = 8/27 := by sorry

end NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l1534_153462


namespace NUMINAMATH_CALUDE_greatest_constant_inequality_l1534_153493

theorem greatest_constant_inequality (α : ℝ) (hα : α > 0) :
  ∃ C : ℝ, (C = 8) ∧ 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y + y * z + z * x = α →
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ C * (x / z + z / x + 2)) ∧
  (∀ C' : ℝ, C' > C →
    ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = α ∧
      (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) < C' * (x / z + z / x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_constant_inequality_l1534_153493


namespace NUMINAMATH_CALUDE_boat_speed_l1534_153497

/-- The speed of a boat in still water, given its speeds with and against the stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 16) 
  (h2 : against_stream = 6) : 
  (along_stream + against_stream) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1534_153497


namespace NUMINAMATH_CALUDE_work_hours_calculation_l1534_153474

/-- Calculates the number of hours spent at work given the total hours in a day and the percentage of time spent working. -/
def hours_at_work (total_hours : ℝ) (work_percentage : ℝ) : ℝ :=
  total_hours * work_percentage

/-- Proves that given a 16-hour day where 50% is spent at work, the number of hours spent at work is 8. -/
theorem work_hours_calculation (total_hours : ℝ) (work_percentage : ℝ) 
    (h1 : total_hours = 16) 
    (h2 : work_percentage = 0.5) : 
  hours_at_work total_hours work_percentage = 8 := by
  sorry

#eval hours_at_work 16 0.5

end NUMINAMATH_CALUDE_work_hours_calculation_l1534_153474


namespace NUMINAMATH_CALUDE_smallest_quotient_three_digit_number_l1534_153424

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_quotient_three_digit_number :
  ∀ n : ℕ, is_three_digit_number n →
    (n : ℚ) / (digit_sum n : ℚ) ≥ 199 / 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_quotient_three_digit_number_l1534_153424


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1534_153465

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem binomial_expansion_coefficient (a : ℚ) : 
  (binomial_coefficient 6 3 : ℚ) * a^3 = 5/2 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1534_153465


namespace NUMINAMATH_CALUDE_village_population_l1534_153432

/-- Given that 90% of a population is 23040, prove that the total population is 25600 -/
theorem village_population (population : ℕ) : 
  (90 : ℚ) / 100 * population = 23040 → population = 25600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1534_153432


namespace NUMINAMATH_CALUDE_f_and_g_increasing_l1534_153454

-- Define the functions
def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem f_and_g_increasing :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → g x < g y) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_increasing_l1534_153454


namespace NUMINAMATH_CALUDE_ratio_equality_l1534_153495

theorem ratio_equality : (2024^2 - 2017^2) / (2031^2 - 2010^2) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l1534_153495


namespace NUMINAMATH_CALUDE_simplify_fraction_l1534_153433

theorem simplify_fraction : (222 : ℚ) / 8888 * 44 = 111 / 101 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1534_153433


namespace NUMINAMATH_CALUDE_books_read_difference_l1534_153463

/-- Given a collection of books and the percentages read by different people,
    calculate the difference between Peter's read books and the combined total of others. -/
theorem books_read_difference (total_books : ℕ) 
  (peter_percent : ℚ) (brother_percent : ℚ) (sarah_percent : ℚ) (alex_percent : ℚ) :
  total_books = 80 →
  peter_percent = 70 / 100 →
  brother_percent = 35 / 100 →
  sarah_percent = 40 / 100 →
  alex_percent = 22 / 100 →
  (peter_percent * total_books : ℚ).floor - 
  ((brother_percent * total_books : ℚ).floor + 
   (sarah_percent * total_books : ℚ).floor + 
   (alex_percent * total_books : ℚ).ceil) = -22 := by
  sorry

#check books_read_difference

end NUMINAMATH_CALUDE_books_read_difference_l1534_153463


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l1534_153426

theorem intersection_point_k_value (k : ℝ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 + k * p.2 = 0) ∧ 
    (2 * p.1 + 3 * p.2 + 8 = 0) ∧ 
    (p.1 - p.2 - 1 = 0)) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l1534_153426


namespace NUMINAMATH_CALUDE_circle_regions_theorem_l1534_153436

/-- Represents the areas of regions in a circle circumscribed around a right triangle -/
structure CircleRegions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides of the right triangle -/
def triangle_sides : (ℝ × ℝ × ℝ) := (15, 20, 25)

/-- The circle is circumscribed around the triangle -/
axiom is_circumscribed (r : CircleRegions) : True

/-- C is the largest region (semicircle) -/
axiom C_is_largest (r : CircleRegions) : r.C ≥ r.A ∧ r.C ≥ r.B

/-- The area of the triangle -/
def triangle_area : ℝ := 150

/-- The theorem to prove -/
theorem circle_regions_theorem (r : CircleRegions) : 
  r.A + r.B + triangle_area = r.C :=
sorry

end NUMINAMATH_CALUDE_circle_regions_theorem_l1534_153436


namespace NUMINAMATH_CALUDE_center_sum_is_ten_l1534_153438

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def center (c : Circle) : ℝ × ℝ := sorry

/-- The sum of the coordinates of a point -/
def coord_sum (p : ℝ × ℝ) : ℝ := p.1 + p.2

/-- The specific circle from the problem -/
def problem_circle : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 6*x - 14*y + 24 = 0 }

theorem center_sum_is_ten :
  coord_sum (center problem_circle) = 10 := by sorry

end NUMINAMATH_CALUDE_center_sum_is_ten_l1534_153438


namespace NUMINAMATH_CALUDE_work_completion_time_l1534_153409

theorem work_completion_time (a b c : ℝ) (h1 : b = 5) (h2 : c = 12) 
  (h3 : 1 / a + 1 / b + 1 / c = 9 / 10) : a = 60 / 37 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1534_153409


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l1534_153445

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 1.25% -/
theorem simple_interest_rate_problem :
  let principal : ℚ := 750
  let amount : ℚ := 900
  let time : ℕ := 16
  simple_interest_rate principal amount time = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l1534_153445


namespace NUMINAMATH_CALUDE_dragon_disc_reassembly_l1534_153443

/-- A circular disc with a dragon painted on it -/
structure DragonDisc where
  center : ℝ × ℝ
  radius : ℝ
  dragon_center : ℝ × ℝ

/-- Two discs are congruent if they have the same radius -/
def congruent (d1 d2 : DragonDisc) : Prop := d1.radius = d2.radius

/-- The dragon covers the center of the disc if its center coincides with the disc's center -/
def dragon_covers_center (d : DragonDisc) : Prop := d.center = d.dragon_center

/-- A disc can be cut and reassembled if there exists a line that divides it into two pieces -/
def can_cut_and_reassemble (d : DragonDisc) : Prop := 
  ∃ (line : ℝ × ℝ → ℝ × ℝ → Prop), 
    ∃ (piece1 piece2 : Set (ℝ × ℝ)), 
      piece1 ∪ piece2 = {p | (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2}

theorem dragon_disc_reassembly 
  (d1 d2 : DragonDisc)
  (h_congruent : congruent d1 d2)
  (h_d1_center : dragon_covers_center d1)
  (h_d2_offset : ¬dragon_covers_center d2) :
  can_cut_and_reassemble d2 ∧ 
  ∃ (d2_new : DragonDisc), congruent d2 d2_new ∧ dragon_covers_center d2_new :=
by sorry

end NUMINAMATH_CALUDE_dragon_disc_reassembly_l1534_153443


namespace NUMINAMATH_CALUDE_rhombus_diagonal_and_side_l1534_153472

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rhombus -/
structure Rhombus where
  A : Point
  C : Point
  AB : Line

/-- Theorem about the diagonals and sides of a specific rhombus -/
theorem rhombus_diagonal_and_side 
  (ABCD : Rhombus)
  (h1 : ABCD.A = ⟨0, 2⟩)
  (h2 : ABCD.C = ⟨4, 6⟩)
  (h3 : ABCD.AB = ⟨3, -1, 2⟩) :
  ∃ (BD AD : Line),
    BD = ⟨1, 1, -6⟩ ∧
    AD = ⟨1, -3, 14⟩ := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_and_side_l1534_153472


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l1534_153460

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 360 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l1534_153460


namespace NUMINAMATH_CALUDE_campsite_tent_ratio_l1534_153471

/-- Represents the number of tents in different areas of a campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  south : ℕ
  center : ℕ
  total : ℕ

/-- The ratio of center tents to north tents is 4:1 given the campsite conditions -/
theorem campsite_tent_ratio (c : CampsiteTents) 
  (h1 : c.total = 900)
  (h2 : c.north = 100)
  (h3 : c.east = 2 * c.north)
  (h4 : c.south = 200)
  (h5 : c.total = c.north + c.east + c.south + c.center) :
  c.center / c.north = 4 := by
  sorry

#check campsite_tent_ratio

end NUMINAMATH_CALUDE_campsite_tent_ratio_l1534_153471


namespace NUMINAMATH_CALUDE_officer_selection_count_l1534_153422

/-- Represents the number of members in the Robotics Club -/
def total_members : ℕ := 24

/-- Represents the number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- Represents the number of constrained pairs (Rachel and Samuel, Tim and Uma) -/
def constrained_pairs : ℕ := 2

/-- Calculates the number of ways to select officers given the constraints -/
def select_officers : ℕ := sorry

/-- Theorem stating that the number of ways to select officers is 126424 -/
theorem officer_selection_count :
  select_officers = 126424 := by sorry

end NUMINAMATH_CALUDE_officer_selection_count_l1534_153422


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l1534_153406

/-- Given a student with homework pages and total problems, prove the number of problems per page. -/
theorem homework_problems_per_page (math_pages reading_pages total_pages total_problems : ℕ)
  (h1 : math_pages + reading_pages = total_pages)
  (h2 : total_pages * (total_problems / total_pages) = total_problems)
  (h3 : total_pages > 0)
  : total_problems / total_pages = 5 :=
by sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l1534_153406


namespace NUMINAMATH_CALUDE_equidistant_line_theorem_l1534_153419

-- Define the points
def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -5)

-- Define the property of the line being equidistant from A and B
def is_equidistant (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), l x y → (abs (3 * x + 2 * y - 7) = abs (4 * x + y - 6))

-- Define the two possible line equations
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 7 = 0
def line2 (x y : ℝ) : Prop := 4 * x + y - 6 = 0

-- Theorem statement
theorem equidistant_line_theorem :
  ∃ (l : ℝ → ℝ → Prop), 
    (l P.1 P.2) ∧ 
    (is_equidistant l) ∧ 
    (∀ (x y : ℝ), l x y ↔ (line1 x y ∨ line2 x y)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_line_theorem_l1534_153419


namespace NUMINAMATH_CALUDE_skew_and_parallel_imply_not_parallel_l1534_153484

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not coplanar and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- Two lines are parallel if they have the same direction vector -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

theorem skew_and_parallel_imply_not_parallel (a b c : Line3D) :
  are_skew a b → are_parallel a c → ¬ are_parallel b c := by
  sorry

end NUMINAMATH_CALUDE_skew_and_parallel_imply_not_parallel_l1534_153484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1534_153489

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence satisfying
    the given condition, 2a_7 - a_8 equals 24. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_condition : a 3 + 3 * a 6 + a 9 = 120) : 
  2 * a 7 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1534_153489


namespace NUMINAMATH_CALUDE_angle_terminal_side_theorem_l1534_153420

theorem angle_terminal_side_theorem (θ : Real) :
  let P : ℝ × ℝ := (-4, 3)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  (∃ t : ℝ, t > 0 ∧ t * (Real.cos θ) = P.1 ∧ t * (Real.sin θ) = P.2) →
  3 * Real.sin θ + Real.cos θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_theorem_l1534_153420


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1534_153403

theorem pentagon_rectangle_ratio : 
  let pentagon_perimeter : ℝ := 60
  let rectangle_perimeter : ℝ := 60
  let pentagon_side : ℝ := pentagon_perimeter / 5
  let rectangle_width : ℝ := rectangle_perimeter / 6
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1534_153403


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1534_153421

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1534_153421


namespace NUMINAMATH_CALUDE_cylinder_volume_from_unfolded_surface_l1534_153453

/-- 
Given a cylinder whose lateral surface unfolds into a square with side length 1,
the volume of the cylinder is 1/(4π).
-/
theorem cylinder_volume_from_unfolded_surface (r h : ℝ) : 
  (2 * π * r = 1) → (h = 1) → (π * r^2 * h = 1 / (4 * π)) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_unfolded_surface_l1534_153453


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt5_l1534_153416

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_magnitude_equals_sqrt5 : 
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt5_l1534_153416


namespace NUMINAMATH_CALUDE_no_real_solutions_l1534_153428

theorem no_real_solutions : ¬ ∃ x : ℝ, (x + 8)^2 = -|x| - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1534_153428


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1534_153404

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1534_153404


namespace NUMINAMATH_CALUDE_impossible_table_filling_l1534_153444

/-- Represents a table filled with digits -/
def Table := Matrix (Fin 5) (Fin 8) Nat

/-- Checks if a digit appears in exactly four rows of the table -/
def appearsInFourRows (t : Table) (d : Nat) : Prop :=
  (Finset.filter (fun i => ∃ j, t i j = d) Finset.univ).card = 4

/-- Checks if a digit appears in exactly four columns of the table -/
def appearsInFourCols (t : Table) (d : Nat) : Prop :=
  (Finset.filter (fun j => ∃ i, t i j = d) Finset.univ).card = 4

/-- A valid table satisfies the conditions for all digits -/
def isValidTable (t : Table) : Prop :=
  ∀ d, d ≤ 9 → appearsInFourRows t d ∧ appearsInFourCols t d

theorem impossible_table_filling : ¬ ∃ t : Table, isValidTable t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_filling_l1534_153444


namespace NUMINAMATH_CALUDE_pugsley_has_four_spiders_l1534_153441

/-- The number of spiders Pugsley has before trading -/
def pugsley_spiders : ℕ := sorry

/-- The number of spiders Wednesday has before trading -/
def wednesday_spiders : ℕ := sorry

/-- Condition 1: If Pugsley gives Wednesday 2 spiders, Wednesday would have 9 times as many spiders as Pugsley -/
axiom condition1 : wednesday_spiders + 2 = 9 * (pugsley_spiders - 2)

/-- Condition 2: If Wednesday gives Pugsley 6 spiders, Pugsley would have 6 fewer spiders than Wednesday had before they traded -/
axiom condition2 : pugsley_spiders + 6 = wednesday_spiders - 6

/-- Theorem: Pugsley has 4 spiders before the trading game commences -/
theorem pugsley_has_four_spiders : pugsley_spiders = 4 := by sorry

end NUMINAMATH_CALUDE_pugsley_has_four_spiders_l1534_153441


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1534_153468

/-- The quadratic equation (k+2)x^2 + 4x + 1 = 0 has two distinct real roots if and only if k < 2 and k ≠ -2 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k + 2) * x^2 + 4 * x + 1 = 0 ∧ (k + 2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1534_153468


namespace NUMINAMATH_CALUDE_pear_juice_percentage_is_correct_l1534_153494

/-- The amount of pear juice produced by a single pear -/
def pear_juice_per_fruit : ℚ := 10 / 5

/-- The amount of orange juice produced by a single orange -/
def orange_juice_per_fruit : ℚ := 12 / 3

/-- The number of each fruit used in the blend -/
def fruits_in_blend : ℕ := 10

/-- The total amount of pear juice in the blend -/
def pear_juice_in_blend : ℚ := fruits_in_blend * pear_juice_per_fruit

/-- The total amount of orange juice in the blend -/
def orange_juice_in_blend : ℚ := fruits_in_blend * orange_juice_per_fruit

/-- The total amount of juice in the blend -/
def total_juice_in_blend : ℚ := pear_juice_in_blend + orange_juice_in_blend

/-- The percentage of pear juice in the blend -/
def pear_juice_percentage : ℚ := pear_juice_in_blend / total_juice_in_blend * 100

theorem pear_juice_percentage_is_correct : 
  pear_juice_percentage = 100/3 := by sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_is_correct_l1534_153494


namespace NUMINAMATH_CALUDE_max_value_of_a_l1534_153429

open Real

theorem max_value_of_a (e : ℝ) (h_e : e = exp 1) :
  (∀ x ∈ Set.Icc (1/e) 2, (a + e) * x - 1 - log x ≤ 0) →
  a ≤ -e :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1534_153429


namespace NUMINAMATH_CALUDE_cos_double_angle_tan_l1534_153442

theorem cos_double_angle_tan (θ : Real) (h : Real.tan θ = -1/3) : 
  Real.cos (2 * θ) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_double_angle_tan_l1534_153442


namespace NUMINAMATH_CALUDE_spelling_contest_questions_l1534_153413

/-- Represents the number of questions in a spelling contest -/
structure SpellingContest where
  drew_correct : Nat
  drew_wrong : Nat
  carla_correct : Nat
  carla_wrong : Nat

/-- The total number of questions in the spelling contest -/
def total_questions (contest : SpellingContest) : Nat :=
  contest.drew_correct + contest.drew_wrong + contest.carla_correct + contest.carla_wrong

/-- Theorem stating the total number of questions in the given spelling contest -/
theorem spelling_contest_questions : ∃ (contest : SpellingContest),
  contest.drew_correct = 20 ∧
  contest.drew_wrong = 6 ∧
  contest.carla_correct = 14 ∧
  contest.carla_wrong = 2 * contest.drew_wrong ∧
  total_questions contest = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_l1534_153413


namespace NUMINAMATH_CALUDE_smallest_greater_than_1_1_l1534_153470

def given_set : Set ℚ := {1.4, 9/10, 1.2, 0.5, 13/10}

theorem smallest_greater_than_1_1 :
  ∃ x ∈ given_set, x > 1.1 ∧ ∀ y ∈ given_set, y > 1.1 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_greater_than_1_1_l1534_153470


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1534_153456

theorem inequality_system_solution (m : ℝ) : 
  (0 ≤ m ∧ m < 1) →
  (∃! (x : ℕ), x > 0 ∧ x - m > 0 ∧ x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1534_153456


namespace NUMINAMATH_CALUDE_cube_surface_area_l1534_153450

/-- Given a cube where the sum of all edge lengths is 72 cm, prove its surface area is 216 cm² -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 72) : 
  let edge_length := edge_sum / 12
  6 * edge_length^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1534_153450


namespace NUMINAMATH_CALUDE_coffee_package_size_l1534_153401

/-- Given two types of coffee packages with the following properties:
    - The total amount of coffee is 70 ounces
    - There are two more packages of the first type than the second type
    - There are 4 packages of the second type, each containing 10 ounces
    Prove that the size of the first type of package is 5 ounces. -/
theorem coffee_package_size
  (total_coffee : ℕ)
  (package_type1 : ℕ)
  (package_type2 : ℕ)
  (size_type2 : ℕ)
  (h1 : total_coffee = 70)
  (h2 : package_type1 = package_type2 + 2)
  (h3 : package_type2 = 4)
  (h4 : size_type2 = 10)
  (h5 : package_type1 * (total_coffee - package_type2 * size_type2) / package_type1 = total_coffee - package_type2 * size_type2) :
  (total_coffee - package_type2 * size_type2) / package_type1 = 5 :=
sorry

end NUMINAMATH_CALUDE_coffee_package_size_l1534_153401


namespace NUMINAMATH_CALUDE_solution_set_implies_values_l1534_153452

/-- Given that the solution set of ax^2 + bx + a^2 - 1 ≤ 0 is [-1, +∞), prove a = 0 and b = -1 -/
theorem solution_set_implies_values (a b : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) → 
  a = 0 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_values_l1534_153452


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1534_153480

theorem arithmetic_mean_problem (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1534_153480


namespace NUMINAMATH_CALUDE_parameterization_validity_l1534_153481

def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

def valid_parameterization (p₀ : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_equation (p₀.1 + t * v.1) (p₀.2 + t * v.2)

theorem parameterization_validity :
  valid_parameterization (0, 4) (1, -3) ∧
  valid_parameterization (-2, 10) (-2, 6) ∧
  valid_parameterization (-1, 7) (2, -6) ∧
  ¬ valid_parameterization (1, 1) (3, -1) ∧
  ¬ valid_parameterization (4, -8) (0.5, -1.5) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l1534_153481


namespace NUMINAMATH_CALUDE_time_difference_per_question_l1534_153459

/-- Calculates the difference in time per question between Math and English exams -/
theorem time_difference_per_question 
  (english_questions : ℕ) 
  (math_questions : ℕ)
  (english_time : ℕ) 
  (math_time : ℕ)
  (h1 : english_questions = 50)
  (h2 : math_questions = 20)
  (h3 : english_time = 80)
  (h4 : math_time = 110) : 
  (math_time : ℚ) / math_questions - (english_time : ℚ) / english_questions = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l1534_153459


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l1534_153407

/-- Two similar right triangles with legs 12 and 9 in the first, and x and 6 in the second, have x = 8 -/
theorem similar_triangles_leg_length : ∀ x : ℝ,
  (12 : ℝ) / x = 9 / 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l1534_153407


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1534_153448

def repeating_decimal_23 : ℚ := 23 / 99
def repeating_decimal_056 : ℚ := 56 / 999
def repeating_decimal_004 : ℚ := 4 / 999

theorem sum_of_repeating_decimals :
  repeating_decimal_23 + repeating_decimal_056 + repeating_decimal_004 = 28917 / 98901 ∧
  (∀ n : ℕ, n > 1 → ¬(n ∣ 28917 ∧ n ∣ 98901)) := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1534_153448


namespace NUMINAMATH_CALUDE_texas_tech_sales_proof_l1534_153487

/-- Calculates the money made from t-shirt sales during the Texas Tech game -/
def texas_tech_sales (profit_per_shirt : ℕ) (total_shirts : ℕ) (arkansas_shirts : ℕ) : ℕ :=
  (total_shirts - arkansas_shirts) * profit_per_shirt

/-- Proves that the money made from t-shirt sales during the Texas Tech game is $1092 -/
theorem texas_tech_sales_proof :
  texas_tech_sales 78 186 172 = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_sales_proof_l1534_153487


namespace NUMINAMATH_CALUDE_micheal_licks_proof_l1534_153475

/-- The number of licks it takes for Dan to reach the center of a lollipop. -/
def dan_licks : ℕ := 58

/-- The number of licks it takes for Sam to reach the center of a lollipop. -/
def sam_licks : ℕ := 70

/-- The number of licks it takes for David to reach the center of a lollipop. -/
def david_licks : ℕ := 70

/-- The number of licks it takes for Lance to reach the center of a lollipop. -/
def lance_licks : ℕ := 39

/-- The average number of licks it takes for all 5 people to reach the center of a lollipop. -/
def average_licks : ℕ := 60

/-- The number of licks it takes for Micheal to reach the center of a lollipop. -/
def micheal_licks : ℕ := 63

/-- Theorem stating that Micheal takes 63 licks to reach the center of a lollipop,
    given the number of licks for Dan, Sam, David, Lance, and the average. -/
theorem micheal_licks_proof :
  (dan_licks + sam_licks + david_licks + lance_licks + micheal_licks) / 5 = average_licks :=
by sorry

end NUMINAMATH_CALUDE_micheal_licks_proof_l1534_153475


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l1534_153485

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_15 :
  units_digit (sum_factorials 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l1534_153485


namespace NUMINAMATH_CALUDE_heavy_rain_time_is_14_minutes_l1534_153405

/-- Represents the weather conditions during the trip -/
inductive Weather
  | Sun
  | LightRain
  | HeavyRain

/-- Represents Shelby's scooter journey -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speedSun : ℝ
  speedLightRain : ℝ
  speedHeavyRain : ℝ

/-- Calculates the time spent in heavy rain given a journey -/
def timeInHeavyRain (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that given the specific journey conditions, 
    the time spent in heavy rain is 14 minutes -/
theorem heavy_rain_time_is_14_minutes 
  (j : Journey)
  (h1 : j.totalDistance = 18)
  (h2 : j.totalTime = 50)
  (h3 : j.speedSun = 30)
  (h4 : j.speedLightRain = 20)
  (h5 : j.speedHeavyRain = 15)
  (h6 : timeInHeavyRain j = timeInHeavyRain j) -- Represents equal rain segments
  : timeInHeavyRain j = 14 := by
  sorry

end NUMINAMATH_CALUDE_heavy_rain_time_is_14_minutes_l1534_153405


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1534_153425

theorem smallest_dual_base_representation : ∃ (n : ℕ) (a b : ℕ), 
  a > 3 ∧ b > 3 ∧
  n = a + 3 ∧
  n = 3 * b + 1 ∧
  (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = c + 3 → m = 3 * d + 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1534_153425


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1534_153439

theorem math_club_team_selection (total_boys : Nat) (total_girls : Nat) (team_size : Nat) :
  total_boys = 10 →
  total_girls = 12 →
  team_size = 8 →
  (team_size / 2 : Nat) = 4 →
  Nat.choose total_boys (team_size / 2) * Nat.choose total_girls (team_size / 2) = 103950 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1534_153439


namespace NUMINAMATH_CALUDE_matthew_baking_time_l1534_153458

/-- Represents the time in hours for Matthew's baking process -/
structure BakingTime where
  assembly : ℝ
  normalBaking : ℝ
  decoration : ℝ
  bakingMultiplier : ℝ

/-- Calculates the total time for Matthew's baking process on the day the oven failed -/
def totalBakingTime (bt : BakingTime) : ℝ :=
  bt.assembly + (bt.normalBaking * bt.bakingMultiplier) + bt.decoration

/-- Theorem stating that Matthew's total baking time on the day the oven failed is 5 hours -/
theorem matthew_baking_time :
  ∀ bt : BakingTime,
  bt.assembly = 1 →
  bt.normalBaking = 1.5 →
  bt.decoration = 1 →
  bt.bakingMultiplier = 2 →
  totalBakingTime bt = 5 := by
sorry

end NUMINAMATH_CALUDE_matthew_baking_time_l1534_153458


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1534_153447

theorem gcd_polynomial_and_multiple (a : ℤ) : 
  (∃ k : ℤ, a = 270 * k) → Int.gcd (5*a^3 + 3*a^2 + 5*a + 45) a = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1534_153447


namespace NUMINAMATH_CALUDE_product_mod_800_l1534_153418

theorem product_mod_800 : (2437 * 2987) % 800 = 109 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_800_l1534_153418


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_theorem_l1534_153457

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The focus-to-asymptote distance equals the real axis length -/
def focus_asymptote_condition (h : Hyperbola a b) : Prop :=
  ∃ c, c^2 = a^2 + b^2 ∧ b * c / (a^2 + b^2).sqrt = 2 * a

/-- The equation of the asymptote -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

/-- Theorem: If the focus-to-asymptote distance equals the real axis length,
    then the asymptote equation is y = ±2x -/
theorem hyperbola_asymptote_theorem (a b : ℝ) (h : Hyperbola a b) :
  focus_asymptote_condition h → ∀ x y, asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_theorem_l1534_153457


namespace NUMINAMATH_CALUDE_numbers_with_zero_from_1_to_700_l1534_153464

def count_numbers_with_zero (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem numbers_with_zero_from_1_to_700 :
  count_numbers_with_zero 1 700 = 123 := by sorry

end NUMINAMATH_CALUDE_numbers_with_zero_from_1_to_700_l1534_153464


namespace NUMINAMATH_CALUDE_total_money_is_102_l1534_153400

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def anna_money (jack : ℕ) : ℕ := jack * 2

def total_money (eric ben jack anna : ℕ) : ℕ := eric + ben + jack + anna

theorem total_money_is_102 :
  total_money (eric_money (ben_money jack_money)) (ben_money jack_money) jack_money (anna_money jack_money) = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_102_l1534_153400


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l1534_153486

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c that makes the lines y = 12x + 5 and y = (3c-1)x - 7 parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 12 * x + 5 ↔ y = (3 * c - 1) * x - 7) → c = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l1534_153486


namespace NUMINAMATH_CALUDE_dimes_per_machine_is_100_l1534_153455

/-- Represents the number of machines in the launderette -/
def num_machines : ℕ := 3

/-- Represents the number of quarters in each machine -/
def quarters_per_machine : ℕ := 80

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the total amount of money from all machines in cents -/
def total_money : ℕ := 9000  -- $90 in cents

/-- Calculates the number of dimes in each machine -/
def dimes_per_machine : ℕ :=
  (total_money - num_machines * quarters_per_machine * quarter_value) / (num_machines * dime_value)

/-- Theorem stating that the number of dimes in each machine is 100 -/
theorem dimes_per_machine_is_100 : dimes_per_machine = 100 := by
  sorry

end NUMINAMATH_CALUDE_dimes_per_machine_is_100_l1534_153455


namespace NUMINAMATH_CALUDE_f_cos_x_l1534_153466

theorem f_cos_x (f : ℝ → ℝ) (h : ∀ x, f (Real.sin x) = 3 - Real.cos (2 * x)) :
  ∀ x, f (Real.cos x) = 3 + Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_cos_x_l1534_153466


namespace NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l1534_153437

-- Define the workday duration in minutes
def workday_minutes : ℕ := 8 * 60

-- Define the duration of the first meeting
def first_meeting_duration : ℕ := 30

-- Define the duration of the second meeting
def second_meeting_duration : ℕ := 2 * first_meeting_duration

-- Define the duration of the third meeting
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

-- Define the total time spent in meetings
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

-- Theorem to prove the percentage of workday spent in meetings
theorem percentage_of_workday_in_meetings :
  (total_meeting_time : ℚ) / workday_minutes * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l1534_153437


namespace NUMINAMATH_CALUDE_largest_selected_is_57_l1534_153408

/-- Represents a systematic sampling of a numbered set of elements. -/
structure SystematicSampling where
  total_elements : ℕ
  smallest_selected : ℕ
  second_smallest_selected : ℕ

/-- Calculates the largest selected number in a systematic sampling. -/
def largest_selected (s : SystematicSampling) : ℕ :=
  let interval := s.second_smallest_selected - s.smallest_selected
  let sample_size := s.total_elements / interval
  s.smallest_selected + interval * (sample_size - 1)

/-- Theorem stating that for the given systematic sampling, the largest selected number is 57. -/
theorem largest_selected_is_57 (s : SystematicSampling) 
  (h1 : s.total_elements = 60)
  (h2 : s.smallest_selected = 3)
  (h3 : s.second_smallest_selected = 9) : 
  largest_selected s = 57 := by
  sorry

end NUMINAMATH_CALUDE_largest_selected_is_57_l1534_153408


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1534_153491

theorem pizza_payment_difference :
  -- Define the total number of slices
  let total_slices : ℕ := 12
  -- Define the cost of a plain pizza
  let plain_pizza_cost : ℚ := 12
  -- Define the additional cost for extra cheese
  let extra_cheese_cost : ℚ := 3
  -- Define the number of slices with extra cheese (one-third of the pizza)
  let extra_cheese_slices : ℕ := total_slices / 3
  -- Define the number of plain slices
  let plain_slices : ℕ := total_slices - extra_cheese_slices
  -- Define the total cost of the pizza
  let total_cost : ℚ := plain_pizza_cost + extra_cheese_cost
  -- Define the cost per slice
  let cost_per_slice : ℚ := total_cost / total_slices
  -- Define the number of slices Nancy ate
  let nancy_slices : ℕ := extra_cheese_slices + 3
  -- Define the number of slices Carol ate
  let carol_slices : ℕ := total_slices - nancy_slices
  -- Define Nancy's payment
  let nancy_payment : ℚ := cost_per_slice * nancy_slices
  -- Define Carol's payment
  let carol_payment : ℚ := cost_per_slice * carol_slices
  -- The theorem to prove
  nancy_payment - carol_payment = (5/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l1534_153491


namespace NUMINAMATH_CALUDE_total_overtime_hours_approx_l1534_153430

/-- Calculates the total overtime hours for three workers given their regular pay rates and total payments. -/
def total_overtime_hours (regular_pay_A regular_pay_B regular_pay_C : ℚ) 
  (total_payment_A total_payment_B total_payment_C : ℚ) : ℚ :=
  let max_regular_hours := 40
  let overtime_pay_A := 2 * regular_pay_A
  let overtime_pay_B := 2 * regular_pay_B
  let overtime_pay_C := 2 * regular_pay_C
  let max_regular_payment_A := max_regular_hours * regular_pay_A
  let max_regular_payment_B := max_regular_hours * regular_pay_B
  let max_regular_payment_C := max_regular_hours * regular_pay_C
  let overtime_payment_A := total_payment_A - max_regular_payment_A
  let overtime_payment_B := total_payment_B - max_regular_payment_B
  let overtime_payment_C := total_payment_C - max_regular_payment_C
  let overtime_hours_A := overtime_payment_A / overtime_pay_A
  let overtime_hours_B := overtime_payment_B / overtime_pay_B
  let overtime_hours_C := overtime_payment_C / overtime_pay_C
  overtime_hours_A + overtime_hours_B + overtime_hours_C

/-- The total overtime hours for the three workers is approximately 21.93 hours. -/
theorem total_overtime_hours_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_overtime_hours 5 6 7 280 330 370 - 21.93| < ε :=
sorry

end NUMINAMATH_CALUDE_total_overtime_hours_approx_l1534_153430


namespace NUMINAMATH_CALUDE_blue_box_lightest_l1534_153476

/-- Represents the color of balls -/
inductive BallColor
  | Yellow
  | White
  | Blue

/-- Represents a box of balls -/
structure Box where
  color : BallColor
  ballCount : Nat
  ballWeight : Nat

/-- Calculate the total weight of balls in a box -/
def boxWeight (box : Box) : Nat :=
  box.ballCount * box.ballWeight

/-- Theorem: The box with blue balls has the lightest weight -/
theorem blue_box_lightest (yellowBox whiteBox blueBox : Box)
  (h_yellow : yellowBox.color = BallColor.Yellow ∧ yellowBox.ballCount = 50 ∧ yellowBox.ballWeight = 50)
  (h_white : whiteBox.color = BallColor.White ∧ whiteBox.ballCount = 60 ∧ whiteBox.ballWeight = 45)
  (h_blue : blueBox.color = BallColor.Blue ∧ blueBox.ballCount = 40 ∧ blueBox.ballWeight = 55) :
  boxWeight blueBox < boxWeight yellowBox ∧ boxWeight blueBox < boxWeight whiteBox :=
by sorry

end NUMINAMATH_CALUDE_blue_box_lightest_l1534_153476


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1534_153417

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/y + 1 = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/b + 1 = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1534_153417


namespace NUMINAMATH_CALUDE_monogram_count_l1534_153431

def alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def before_m : List Char := alphabet.take 12
def after_m : List Char := alphabet.drop 13

def is_valid_monogram (first middle last : Char) : Prop :=
  first ∈ before_m ∧ middle = 'M' ∧ last ∈ after_m ∧ first < middle ∧ middle < last

def count_valid_monograms : Nat :=
  (before_m.length) * (after_m.length)

theorem monogram_count :
  count_valid_monograms = 156 :=
by sorry

end NUMINAMATH_CALUDE_monogram_count_l1534_153431


namespace NUMINAMATH_CALUDE_union_of_overlapping_intervals_l1534_153434

open Set

theorem union_of_overlapping_intervals :
  let A : Set ℝ := {x | 1 < x ∧ x < 3}
  let B : Set ℝ := {x | 2 < x ∧ x < 4}
  A ∪ B = {x | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_overlapping_intervals_l1534_153434


namespace NUMINAMATH_CALUDE_zeros_product_gt_e_squared_l1534_153473

/-- Given a function g(x) = ln x - kx with two distinct zeros x₁ and x₂, 
    prove that their product is greater than e^2. -/
theorem zeros_product_gt_e_squared 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_zero₁ : Real.log x₁ = k * x₁) 
  (h_zero₂ : Real.log x₂ = k * x₂) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_gt_e_squared_l1534_153473


namespace NUMINAMATH_CALUDE_intersection_P_M_l1534_153478

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_M_l1534_153478


namespace NUMINAMATH_CALUDE_soccer_team_defenders_l1534_153477

theorem soccer_team_defenders (total_players : ℕ) (goalies : ℕ) (strikers : ℕ) (defenders : ℕ) :
  total_players = 40 →
  goalies = 3 →
  strikers = 7 →
  defenders + goalies + strikers + 2 * defenders = total_players →
  defenders = 10 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_defenders_l1534_153477


namespace NUMINAMATH_CALUDE_perimeter_of_MNO_l1534_153412

/-- A solid right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge_start : ℝ × ℝ × ℝ
  edge_end : ℝ × ℝ × ℝ

/-- Theorem: Perimeter of triangle MNO in a right prism -/
theorem perimeter_of_MNO (prism : RightPrism) 
  (M : Midpoint) (N : Midpoint) (O : Midpoint) : 
  prism.height = 20 → 
  prism.base_side_length = 10 → 
  -- Assumptions about M, N, O being midpoints of specific edges would be added here
  ∃ (perimeter : ℝ), perimeter = 5 * (2 * Real.sqrt 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_MNO_l1534_153412


namespace NUMINAMATH_CALUDE_problem_1_l1534_153427

theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1534_153427


namespace NUMINAMATH_CALUDE_equation_equivalence_l1534_153467

theorem equation_equivalence (a b c : ℝ) :
  (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1534_153467


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l1534_153490

/-- The volume of an open box constructed from a rectangular metal sheet. -/
def boxVolume (sheetLength sheetWidth y : ℝ) : ℝ :=
  (sheetLength - 2*y) * (sheetWidth - 2*y) * y

/-- Theorem stating the volume of the box for the given dimensions -/
theorem volume_of_specific_box (y : ℝ) :
  boxVolume 18 12 y = 4*y^3 - 60*y^2 + 216*y :=
by sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l1534_153490


namespace NUMINAMATH_CALUDE_temperature_range_l1534_153483

/-- Given the highest and lowest temperatures on a certain day in Chengdu,
    prove that the temperature range is between these two values. -/
theorem temperature_range (highest lowest t : ℝ) 
  (h_highest : highest = 29)
  (h_lowest : lowest = 21)
  (h_range : lowest ≤ t ∧ t ≤ highest) : 
  21 ≤ t ∧ t ≤ 29 := by sorry

end NUMINAMATH_CALUDE_temperature_range_l1534_153483


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1534_153410

theorem quadratic_roots_sum_of_squares (α β : ℝ) : 
  (∀ x, x^2 - 7*x + 3 = 0 ↔ x = α ∨ x = β) →
  α^2 + β^2 = 43 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1534_153410


namespace NUMINAMATH_CALUDE_atlantic_call_charge_l1534_153415

/-- Represents the per minute charge of Atlantic Call in dollars -/
def atlantic_charge : ℚ := 1/5

/-- Represents the base rate of United Telephone in dollars -/
def united_base : ℚ := 8

/-- Represents the per minute charge of United Telephone in dollars -/
def united_per_minute : ℚ := 1/4

/-- Represents the base rate of Atlantic Call in dollars -/
def atlantic_base : ℚ := 12

/-- Represents the number of minutes at which the bills are equal -/
def equal_minutes : ℚ := 80

theorem atlantic_call_charge :
  united_base + united_per_minute * equal_minutes = 
  atlantic_base + atlantic_charge * equal_minutes := by
  sorry

#eval atlantic_charge

end NUMINAMATH_CALUDE_atlantic_call_charge_l1534_153415


namespace NUMINAMATH_CALUDE_plant_growth_equation_l1534_153482

/-- Represents the growth pattern of a plant -/
def plant_growth (x : ℕ) : Prop :=
  -- One main stem
  let main_stem := 1
  -- x branches on the main stem
  let branches := x
  -- x small branches on each of the x branches
  let small_branches := x * x
  -- The total number of stems and branches is 73
  main_stem + branches + small_branches = 73

/-- Theorem stating the equation for the plant's growth pattern -/
theorem plant_growth_equation :
  ∃ x : ℕ, plant_growth x ∧ (1 + x + x^2 = 73) :=
sorry

end NUMINAMATH_CALUDE_plant_growth_equation_l1534_153482


namespace NUMINAMATH_CALUDE_red_squares_per_row_is_six_l1534_153402

/-- Represents a colored grid -/
structure ColoredGrid where
  rows : Nat
  columns : Nat
  redRows : Nat
  blueRows : Nat
  greenSquares : Nat

/-- Calculates the number of squares in each red row -/
def redSquaresPerRow (grid : ColoredGrid) : Nat :=
  let totalSquares := grid.rows * grid.columns
  let blueSquares := grid.blueRows * grid.columns
  let redSquares := totalSquares - blueSquares - grid.greenSquares
  redSquares / grid.redRows

/-- Theorem: In the given grid, there are 6 red squares in each red row -/
theorem red_squares_per_row_is_six (grid : ColoredGrid) 
  (h1 : grid.rows = 10)
  (h2 : grid.columns = 15)
  (h3 : grid.redRows = 4)
  (h4 : grid.blueRows = 4)
  (h5 : grid.greenSquares = 66) :
  redSquaresPerRow grid = 6 := by
  sorry

#eval redSquaresPerRow { rows := 10, columns := 15, redRows := 4, blueRows := 4, greenSquares := 66 }

end NUMINAMATH_CALUDE_red_squares_per_row_is_six_l1534_153402


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l1534_153435

/-- Given Carmen's initial number of cats and dogs, and the number of cats given up for adoption,
    prove that she now has 7 more cats than dogs. -/
theorem carmen_pets_difference (initial_cats initial_dogs cats_adopted : ℕ) 
    (h1 : initial_cats = 28)
    (h2 : initial_dogs = 18)
    (h3 : cats_adopted = 3) : 
  initial_cats - cats_adopted - initial_dogs = 7 := by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l1534_153435


namespace NUMINAMATH_CALUDE_absent_students_percentage_l1534_153488

theorem absent_students_percentage
  (total_students : ℕ)
  (num_boys : ℕ)
  (num_girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 120)
  (h2 : num_boys = 72)
  (h3 : num_girls = 48)
  (h4 : boys_absent_fraction = 1 / 8)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = num_boys + num_girls) :
  (boys_absent_fraction * num_boys + girls_absent_fraction * num_girls) / total_students = 7 / 40 := by
  sorry

#eval (7 : ℚ) / 40 * 100 -- To show that 7/40 is equivalent to 17.5%

end NUMINAMATH_CALUDE_absent_students_percentage_l1534_153488


namespace NUMINAMATH_CALUDE_value_of_p_l1534_153469

theorem value_of_p (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_value_of_p_l1534_153469


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_relations_l1534_153414

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular : Line → Plane → Prop)

-- Define the relation for a line lying within a plane
variable (lies_within : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (line_perpendicular : Line → Line → Prop)

theorem perpendicular_line_plane_relations 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : lies_within m α)
  (h3 : plane_perpendicular α β) :
  (line_perpendicular l m → lies_within m β) ∧ 
  (perpendicular m β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_relations_l1534_153414
