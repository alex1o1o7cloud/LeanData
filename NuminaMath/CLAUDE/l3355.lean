import Mathlib

namespace NUMINAMATH_CALUDE_books_sold_in_garage_sale_l3355_335595

theorem books_sold_in_garage_sale 
  (initial_books : ℝ)
  (books_given_to_friend : ℝ)
  (final_books : ℝ)
  (h1 : initial_books = 284.5)
  (h2 : books_given_to_friend = 63.7)
  (h3 : final_books = 112.3) :
  initial_books - books_given_to_friend - final_books = 108.5 :=
by
  sorry

#eval 284.5 - 63.7 - 112.3  -- This should evaluate to 108.5

end NUMINAMATH_CALUDE_books_sold_in_garage_sale_l3355_335595


namespace NUMINAMATH_CALUDE_extreme_point_of_g_l3355_335530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x - 1

noncomputable def g (x : ℝ) : ℝ := x * (f 1 x) + (1/2) * x^2 + 2 * x

def has_unique_extreme_point (h : ℝ → ℝ) (m : ℤ) : Prop :=
  ∃! (x : ℝ), m < x ∧ x < m + 1 ∧ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≤ h x) ∨ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≥ h x)

theorem extreme_point_of_g :
  ∃ m : ℤ, has_unique_extreme_point g m → m = 0 ∨ m = 3 :=
sorry

end NUMINAMATH_CALUDE_extreme_point_of_g_l3355_335530


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l3355_335589

/-- Represents the money redistribution problem among three friends --/
def money_redistribution (a j t : ℕ) : Prop :=
  let a1 := a - 2*(t + j)
  let j1 := 3*j
  let t1 := 3*t
  let a2 := 2*a1
  let j2 := j1 - (a1 + t1)
  let t2 := 2*t1
  let a3 := 2*a2
  let j3 := 2*j2
  let t3 := t2 - (a2 + j2)
  (t = 48) ∧ (t3 = 48) ∧ (a3 + j3 + t3 = 528)

/-- Theorem stating the total amount of money after redistribution --/
theorem total_money_after_redistribution :
  ∃ (a j : ℕ), money_redistribution a j 48 :=
sorry

end NUMINAMATH_CALUDE_total_money_after_redistribution_l3355_335589


namespace NUMINAMATH_CALUDE_arc_minutes_to_degrees_l3355_335526

theorem arc_minutes_to_degrees :
  ∀ (arc_minutes : ℝ) (degrees : ℝ),
  (arc_minutes = 1200) →
  (degrees = 20) →
  (arc_minutes * (1 / 60) = degrees) :=
by
  sorry

end NUMINAMATH_CALUDE_arc_minutes_to_degrees_l3355_335526


namespace NUMINAMATH_CALUDE_solve_equation_l3355_335518

theorem solve_equation (x : ℝ) (h : x + 1 = 2) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3355_335518


namespace NUMINAMATH_CALUDE_four_numbers_product_2002_sum_less_40_l3355_335531

theorem four_numbers_product_2002_sum_less_40 (a b c d : ℕ+) :
  a * b * c * d = 2002 ∧ a + b + c + d < 40 →
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 13 ∧ d = 11) ∨ (a = 1 ∧ b = 14 ∧ c = 13 ∧ d = 11) ∨
  (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨ (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
  (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨ (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
  (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨ (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
  (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨ (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_product_2002_sum_less_40_l3355_335531


namespace NUMINAMATH_CALUDE_noemi_roulette_loss_l3355_335556

/-- Noemi's gambling problem -/
theorem noemi_roulette_loss 
  (initial_amount : ℕ) 
  (final_amount : ℕ) 
  (blackjack_loss : ℕ) 
  (h1 : initial_amount = 1700)
  (h2 : final_amount = 800)
  (h3 : blackjack_loss = 500) :
  initial_amount - final_amount - blackjack_loss = 400 := by
sorry

end NUMINAMATH_CALUDE_noemi_roulette_loss_l3355_335556


namespace NUMINAMATH_CALUDE_simplify_expression_l3355_335543

theorem simplify_expression : 
  (Real.sqrt 392 / Real.sqrt 336) + (Real.sqrt 200 / Real.sqrt 128) + 1 = 41 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3355_335543


namespace NUMINAMATH_CALUDE_total_texts_is_forty_l3355_335501

/-- The number of texts Sydney sent to Allison and Brittney on both days -/
def total_texts (monday_texts_per_person tuesday_texts_per_person : ℕ) : ℕ :=
  2 * (monday_texts_per_person + tuesday_texts_per_person)

/-- Theorem stating that the total number of texts is 40 -/
theorem total_texts_is_forty :
  total_texts 5 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_texts_is_forty_l3355_335501


namespace NUMINAMATH_CALUDE_min_time_is_200_minutes_l3355_335553

/-- Represents the travel problem between two cities -/
structure TravelProblem where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- Calculates the minimum travel time for the given problem -/
def min_travel_time (problem : TravelProblem) : ℝ :=
  sorry

/-- Theorem stating that the minimum travel time for the given problem is 200 minutes -/
theorem min_time_is_200_minutes :
  let problem : TravelProblem := {
    distance := 45,
    num_people := 3,
    num_bicycles := 2,
    cyclist_speed := 15,
    pedestrian_speed := 5
  }
  min_travel_time problem = 200 / 60 := by sorry

end NUMINAMATH_CALUDE_min_time_is_200_minutes_l3355_335553


namespace NUMINAMATH_CALUDE_inequality_proof_l3355_335503

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (3 * x^2 - x) / (1 + x^2) + (3 * y^2 - y) / (1 + y^2) + (3 * z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3355_335503


namespace NUMINAMATH_CALUDE_distance_between_trees_l3355_335562

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 360 →
  num_trees = 31 →
  yard_length / (num_trees - 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3355_335562


namespace NUMINAMATH_CALUDE_rose_count_l3355_335566

theorem rose_count (lilies roses tulips : ℕ) : 
  roses = lilies + 22 →
  roses = tulips - 20 →
  lilies + roses + tulips = 100 →
  roses = 34 := by
sorry

end NUMINAMATH_CALUDE_rose_count_l3355_335566


namespace NUMINAMATH_CALUDE_married_men_fraction_l3355_335540

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h_women_positive : total_women > 0)
  (h_total_positive : total_people > 0)
  (h_single_prob : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women) / total_people = (4 : ℚ) / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3355_335540


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3355_335541

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility (n : ℕ) : 
  (is_divisible_by (n + 3) 12 ∧ 
   is_divisible_by (n + 3) 15 ∧ 
   is_divisible_by (n + 3) 40) →
  (∀ k : ℕ, k < n → 
    ¬(is_divisible_by (k + 3) 12 ∧ 
      is_divisible_by (k + 3) 15 ∧ 
      is_divisible_by (k + 3) 40)) →
  (∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m) →
  ∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3355_335541


namespace NUMINAMATH_CALUDE_slope_of_line_l3355_335563

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l3355_335563


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3355_335591

theorem difference_of_squares_example : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3355_335591


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3355_335549

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 100 * x + 9 * y^2 - 36 * y = 225

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 10.134

/-- Theorem: The distance between the foci of the ellipse defined by the given equation
    is approximately 10.134 -/
theorem ellipse_foci_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧
  (∀ x y : ℝ, ellipse_equation x y → abs (foci_distance - 10.134) < ε) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3355_335549


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3355_335505

/-- Prove that given an initial amount P, after applying 5% interest for the first year
    and 6% interest for the second year, if the final amount is 5565, then P must be 5000. -/
theorem initial_amount_proof (P : ℝ) : 
  P * (1 + 0.05) * (1 + 0.06) = 5565 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3355_335505


namespace NUMINAMATH_CALUDE_feeding_and_trapping_sets_l3355_335542

/-- A set is a feeding set for a sequence if every open subinterval of the set contains infinitely many terms of the sequence. -/
def IsFeeder (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  ∀ a b, a < b → a ∈ s → b ∈ s → Set.Infinite {n : ℕ | seq n ∈ Set.Ioo a b}

/-- A set is a trapping set for a sequence if no infinite subset of the sequence remains outside the set. -/
def IsTrap (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  Set.Finite {n : ℕ | seq n ∉ s}

theorem feeding_and_trapping_sets :
  (∃ seq : ℕ → ℝ, IsFeeder (Set.Icc 0 1) seq ∧ IsFeeder (Set.Icc 2 3) seq) ∧
  (¬ ∃ seq : ℕ → ℝ, IsTrap (Set.Icc 0 1) seq ∧ IsTrap (Set.Icc 2 3) seq) :=
sorry

end NUMINAMATH_CALUDE_feeding_and_trapping_sets_l3355_335542


namespace NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3355_335535

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its four vertices -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A rhombus defined by its four vertices -/
structure Rhombus where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Check if a point is on a line segment defined by two other points -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y)

/-- The set of all possible locations for the fourth vertex of the rhombus -/
def fourthVertexSet (sq : Square) : Set Point :=
  { p : Point | ∃ r : Rhombus,
    isOnSegment r.P sq.A sq.B ∧
    isOnSegment r.Q sq.B sq.C ∧
    isOnSegment r.R sq.A sq.D ∧
    r.S = p }

/-- The area of a set of points in 2D space -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem rhombus_fourth_vertex_area (sq : Square) :
  sq.A = Point.mk 0 0 →
  sq.B = Point.mk 1 0 →
  sq.C = Point.mk 1 1 →
  sq.D = Point.mk 0 1 →
  area (fourthVertexSet sq) = 7/3 :=
by
  sorry

end NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3355_335535


namespace NUMINAMATH_CALUDE_problem_statement_l3355_335508

theorem problem_statement :
  (∀ (a b m : ℝ), (a * m^2 < b * m^2 → a < b) ∧ ¬(a < b → a * m^2 < b * m^2)) ∧
  (¬(∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 - 1 > 0)) ∧
  (∀ (p q : Prop), ¬p → ¬q → ¬(p ∧ q)) ∧
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ -1 → x^2 ≠ 1) ↔ (x^2 = 1 → x = 1 ∨ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3355_335508


namespace NUMINAMATH_CALUDE_problem_solution_l3355_335527

/-- The number of problems completed given the rate and time -/
def problems_completed (p t : ℕ) : ℕ := p * t

/-- The condition that my friend's completion matches mine -/
def friend_completion_matches (p t : ℕ) : Prop :=
  p * t = (2 * p - 6) * (t - 3)

theorem problem_solution (p t : ℕ) 
  (h1 : p > 15) 
  (h2 : t > 3)
  (h3 : friend_completion_matches p t) :
  problems_completed p t = 216 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3355_335527


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l3355_335502

/-- Given a parabola y = ax² + bx + c with 0 < 2a < b, and points A(1/2, y₁), B(0, y₂), C(-1, y₃) on the parabola,
    prove that y₁ > y₂ > y₃ -/
theorem parabola_point_ordering (a b c y₁ y₂ y₃ : ℝ) :
  0 < 2 * a → 2 * a < b →
  y₁ = a * (1/2)^2 + b * (1/2) + c →
  y₂ = c →
  y₃ = a * (-1)^2 + b * (-1) + c →
  y₁ > y₂ ∧ y₂ > y₃ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l3355_335502


namespace NUMINAMATH_CALUDE_bushes_for_zucchinis_l3355_335571

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 8

/-- Represents the number of containers traded for zucchinis -/
def containers_traded : ℕ := 5

/-- Represents the number of zucchinis received in trade -/
def zucchinis_received : ℕ := 2

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 48

/-- Calculates the number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ :=
  (target_zucchinis * containers_traded) / (zucchinis_received * containers_per_bush)

theorem bushes_for_zucchinis :
  bushes_needed = 15 :=
sorry

end NUMINAMATH_CALUDE_bushes_for_zucchinis_l3355_335571


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3355_335590

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3355_335590


namespace NUMINAMATH_CALUDE_max_selection_ways_l3355_335593

/-- The total number of socks -/
def total_socks : ℕ := 2017

/-- The function to calculate the number of ways to select socks -/
def selection_ways (partition : List ℕ) : ℕ :=
  partition.prod

/-- The theorem stating the maximum number of ways to select socks -/
theorem max_selection_ways :
  ∃ (partition : List ℕ),
    partition.sum = total_socks ∧
    ∀ (other_partition : List ℕ),
      other_partition.sum = total_socks →
      selection_ways other_partition ≤ selection_ways partition ∧
      selection_ways partition = 3^671 * 4 :=
sorry

end NUMINAMATH_CALUDE_max_selection_ways_l3355_335593


namespace NUMINAMATH_CALUDE_geometric_sequence_l3355_335525

theorem geometric_sequence (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) →
  a 1 + a 3 = 10 →
  a 2 + a 4 = 5 →
  a 8 = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_l3355_335525


namespace NUMINAMATH_CALUDE_bank_interest_equation_l3355_335569

theorem bank_interest_equation (initial_deposit : ℝ) (interest_tax_rate : ℝ) 
  (total_amount : ℝ) (annual_interest_rate : ℝ) 
  (h1 : initial_deposit = 2500)
  (h2 : interest_tax_rate = 0.2)
  (h3 : total_amount = 2650) :
  initial_deposit * (1 + annual_interest_rate * (1 - interest_tax_rate)) = total_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_interest_equation_l3355_335569


namespace NUMINAMATH_CALUDE_hyperbola_cosh_sinh_l3355_335575

theorem hyperbola_cosh_sinh (t : ℝ) : (Real.cosh t)^2 - (Real.sinh t)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_cosh_sinh_l3355_335575


namespace NUMINAMATH_CALUDE_jane_well_days_l3355_335582

/-- Represents Jane's performance levels --/
inductive Performance
  | Poor
  | Well
  | Excellent

/-- Returns the daily earnings based on performance --/
def dailyEarnings (p : Performance) : ℕ :=
  match p with
  | Performance.Poor => 2
  | Performance.Well => 4
  | Performance.Excellent => 6

/-- Represents Jane's work record over 15 days --/
structure WorkRecord :=
  (poorDays : ℕ)
  (wellDays : ℕ)
  (excellentDays : ℕ)
  (total_days : poorDays + wellDays + excellentDays = 15)
  (excellent_poor_relation : excellentDays = poorDays + 4)
  (total_earnings : poorDays * 2 + wellDays * 4 + excellentDays * 6 = 66)

/-- Theorem stating that Jane performed well for 11 days --/
theorem jane_well_days (record : WorkRecord) : record.wellDays = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_well_days_l3355_335582


namespace NUMINAMATH_CALUDE_syrup_volume_in_tank_syrup_volume_specific_l3355_335560

/-- The volume of syrup in a partially filled cylindrical tank -/
theorem syrup_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) 
  (fill_ratio : ℝ) (syrup_ratio : ℝ) : ℝ :=
  let tank_radius : ℝ := tank_diameter / 2
  let liquid_height : ℝ := fill_ratio * tank_height
  let liquid_volume : ℝ := Real.pi * tank_radius^2 * liquid_height
  let syrup_volume : ℝ := liquid_volume * syrup_ratio / (1 + 1/syrup_ratio)
  syrup_volume

/-- The volume of syrup in the specific tank described in the problem -/
theorem syrup_volume_specific : 
  ∃ (ε : ℝ), abs (syrup_volume_in_tank 9 4 (1/3) (1/5) - 6.28) < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_syrup_volume_in_tank_syrup_volume_specific_l3355_335560


namespace NUMINAMATH_CALUDE_square_midpoint_dot_product_l3355_335506

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CD := (D.1 - C.1, D.2 - C.2)
  let DA := (A.1 - D.1, A.2 - D.2)
  (AB.1 * AB.1 + AB.2 * AB.2 = 4) ∧
  (BC.1 * BC.1 + BC.2 * BC.2 = 4) ∧
  (CD.1 * CD.1 + CD.2 * CD.2 = 4) ∧
  (DA.1 * DA.1 + DA.2 * DA.2 = 4) ∧
  (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∧
  (BC.1 * CD.1 + BC.2 * CD.2 = 0) ∧
  (CD.1 * DA.1 + CD.2 * DA.2 = 0) ∧
  (DA.1 * AB.1 + DA.2 * AB.2 = 0)

-- Define the midpoint E of CD
def Midpoint (C D E : ℝ × ℝ) : Prop :=
  E.1 = (C.1 + D.1) / 2 ∧ E.2 = (C.2 + D.2) / 2

-- Define the dot product of two vectors
def DotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem square_midpoint_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : Midpoint C D E) : 
  DotProduct (E.1 - A.1, E.2 - A.2) (D.1 - B.1, D.2 - B.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_midpoint_dot_product_l3355_335506


namespace NUMINAMATH_CALUDE_sarah_meal_options_l3355_335528

/-- The number of distinct meals Sarah can order -/
def total_meals (main_courses sides drinks desserts : ℕ) : ℕ :=
  main_courses * sides * drinks * desserts

/-- Theorem stating that Sarah can order 48 distinct meals -/
theorem sarah_meal_options : total_meals 4 3 2 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sarah_meal_options_l3355_335528


namespace NUMINAMATH_CALUDE_chinese_table_tennis_team_arrangements_l3355_335539

/-- The number of players in the Chinese men's table tennis team -/
def total_players : ℕ := 6

/-- The number of players required for the team event -/
def team_size : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

/-- The main theorem -/
theorem chinese_table_tennis_team_arrangements :
  permutations total_players team_size - permutations (total_players - 1) (team_size - 1) = 100 := by
  sorry


end NUMINAMATH_CALUDE_chinese_table_tennis_team_arrangements_l3355_335539


namespace NUMINAMATH_CALUDE_spring_length_increase_l3355_335533

def spring_length (x : ℝ) : ℝ :=
  20 + 0.5 * x

theorem spring_length_increase (x₁ x₂ : ℝ) 
  (h1 : 0 ≤ x₁ ∧ x₁ < 5) 
  (h2 : x₂ = x₁ + 1) : 
  spring_length x₂ - spring_length x₁ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_length_increase_l3355_335533


namespace NUMINAMATH_CALUDE_a_range_l3355_335515

/-- The inequality holds for all positive real x -/
def inequality_holds (a : ℝ) : Prop :=
  ∀ x > 0, a * Real.log (a * x) ≤ Real.exp x

/-- The theorem stating the range of a given the inequality -/
theorem a_range (a : ℝ) (h : inequality_holds a) : 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3355_335515


namespace NUMINAMATH_CALUDE_money_difference_l3355_335537

/-- Given that Bob has $60, Phil has 1/3 of Bob's amount, and Jenna has twice Phil's amount,
    prove that the difference between Bob's and Jenna's amounts is $20. -/
theorem money_difference (bob_amount : ℕ) (phil_amount : ℕ) (jenna_amount : ℕ)
    (h1 : bob_amount = 60)
    (h2 : phil_amount = bob_amount / 3)
    (h3 : jenna_amount = 2 * phil_amount) :
    bob_amount - jenna_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3355_335537


namespace NUMINAMATH_CALUDE_count_rational_coefficient_terms_l3355_335534

/-- The number of terms with rational coefficients in the expansion of (x⁴√2 + y⁵√3)^1200 -/
def rational_coefficient_terms : ℕ :=
  let n : ℕ := 1200
  let f (k : ℕ) : Bool := k % 4 = 0 ∧ (n - k) % 5 = 0
  (List.range (n + 1)).filter f |>.length

theorem count_rational_coefficient_terms : 
  rational_coefficient_terms = 61 := by sorry

end NUMINAMATH_CALUDE_count_rational_coefficient_terms_l3355_335534


namespace NUMINAMATH_CALUDE_diameter_endpoint_theorem_l3355_335538

/-- A circle in a 2D coordinate plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- The theorem stating the relationship between the center and endpoints of a diameter --/
theorem diameter_endpoint_theorem (c : Circle) (d : Diameter) :
  c.center = (5, 2) ∧ d.circle = c ∧ d.endpoint1 = (0, -3) →
  d.endpoint2 = (10, 7) := by
  sorry

end NUMINAMATH_CALUDE_diameter_endpoint_theorem_l3355_335538


namespace NUMINAMATH_CALUDE_largest_digit_rounding_l3355_335583

def number (d : ℕ) : ℕ := 5400000000 + d * 10000000 + 9607502

def rounds_to_5_5_billion (n : ℕ) : Prop :=
  5450000000 ≤ n ∧ n < 5550000000

theorem largest_digit_rounding :
  ∀ d : ℕ, d ≤ 9 →
    (rounds_to_5_5_billion (number d) ↔ 5 ≤ d) ∧
    (d = 9 ↔ ∀ k : ℕ, k ≤ 9 ∧ rounds_to_5_5_billion (number k) → k ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_rounding_l3355_335583


namespace NUMINAMATH_CALUDE_no_integer_solution_l3355_335536

theorem no_integer_solution :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3355_335536


namespace NUMINAMATH_CALUDE_problem_distribution_l3355_335551

def distribute_problems (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k

theorem problem_distribution :
  distribute_problems 9 7 = 181440 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l3355_335551


namespace NUMINAMATH_CALUDE_fraction_division_calculate_fraction_l3355_335574

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c := by
  sorry

theorem calculate_fraction :
  7 / (9 / 14) = 98 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_calculate_fraction_l3355_335574


namespace NUMINAMATH_CALUDE_brown_class_points_l3355_335599

theorem brown_class_points (william_points mr_adams_points daniel_points : ℕ)
  (mean_points : ℚ) (total_classes : ℕ) :
  william_points = 50 →
  mr_adams_points = 57 →
  daniel_points = 57 →
  mean_points = 53.3 →
  total_classes = 4 →
  ∃ (brown_points : ℕ),
    (william_points + mr_adams_points + daniel_points + brown_points) / total_classes = mean_points ∧
    brown_points = 49 :=
by sorry

end NUMINAMATH_CALUDE_brown_class_points_l3355_335599


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l3355_335523

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l3355_335523


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3355_335511

-- Define the function f(x) = x + 2
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem sum_of_max_and_min_is_eight :
  let a : ℝ := 0
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ f b) ∧ 
  (∀ x ∈ Set.Icc a b, f a ≤ f x) →
  f a + f b = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3355_335511


namespace NUMINAMATH_CALUDE_paint_remaining_l3355_335584

theorem paint_remaining (initial_paint : ℚ) : initial_paint = 2 →
  let day1_remaining := initial_paint / 2
  let day2_remaining := day1_remaining * 3 / 4
  let day3_remaining := day2_remaining * 2 / 3
  day3_remaining = initial_paint / 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l3355_335584


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3355_335519

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3355_335519


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3355_335598

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 16) →
  x = 15/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3355_335598


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3355_335532

/-- The curve y = x^3 + ax + 1 passes through the point (2, 3) -/
def curve_passes_through (a : ℝ) : Prop :=
  2^3 + a*2 + 1 = 3

/-- The derivative of the curve y = x^3 + ax + 1 -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ :=
  3*x^2 + a

/-- The line y = kx + b is tangent to the curve y = x^3 + ax + 1 at x = 2 -/
def line_tangent_to_curve (a k b : ℝ) : Prop :=
  k = curve_derivative a 2

/-- The line y = kx + b passes through the point (2, 3) -/
def line_passes_through (k b : ℝ) : Prop :=
  k*2 + b = 3

theorem tangent_line_b_value (a k b : ℝ) :
  curve_passes_through a →
  line_tangent_to_curve a k b →
  line_passes_through k b →
  b = -15 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l3355_335532


namespace NUMINAMATH_CALUDE_fraction_equality_l3355_335512

theorem fraction_equality (a b c x : ℝ) (hx : x = a / b) (hc : c ≠ 0) (hb : b ≠ 0) (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3355_335512


namespace NUMINAMATH_CALUDE_max_mice_caught_max_mice_achievable_l3355_335579

/-- Production Possibility Frontier for a male kitten -/
def male_ppf (k : ℝ) : ℝ := 80 - 4 * k

/-- Production Possibility Frontier for a female kitten -/
def female_ppf (k : ℝ) : ℝ := 16 - 0.25 * k

/-- The maximum number of mice that can be caught by any combination of two kittens -/
def max_mice : ℝ := 160

/-- Theorem stating that the maximum number of mice that can be caught by any combination
    of two kittens is 160 -/
theorem max_mice_caught :
  ∀ k₁ k₂ : ℝ, k₁ ≥ 0 → k₂ ≥ 0 →
  (male_ppf k₁ + male_ppf k₂ ≤ max_mice) ∧
  (male_ppf k₁ + female_ppf k₂ ≤ max_mice) ∧
  (female_ppf k₁ + female_ppf k₂ ≤ max_mice) :=
sorry

/-- Theorem stating that there exist values of k₁ and k₂ for which the maximum is achieved -/
theorem max_mice_achievable :
  ∃ k₁ k₂ : ℝ, k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ male_ppf k₁ + male_ppf k₂ = max_mice :=
sorry

end NUMINAMATH_CALUDE_max_mice_caught_max_mice_achievable_l3355_335579


namespace NUMINAMATH_CALUDE_scientific_notation_of_ten_billion_thirty_million_l3355_335587

theorem scientific_notation_of_ten_billion_thirty_million :
  (10030000000 : ℝ) = 1.003 * (10 : ℝ)^10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_ten_billion_thirty_million_l3355_335587


namespace NUMINAMATH_CALUDE_teachers_separation_probability_l3355_335581

/-- The number of students in the group photo arrangement. -/
def num_students : ℕ := 5

/-- The number of teachers in the group photo arrangement. -/
def num_teachers : ℕ := 2

/-- The total number of people in the group photo arrangement. -/
def total_people : ℕ := num_students + num_teachers

/-- The probability of arranging the group such that the two teachers
    are not at the ends and not adjacent to each other. -/
def probability_teachers_separated : ℚ :=
  (num_students.factorial * (num_students + 1).choose 2) / total_people.factorial

theorem teachers_separation_probability :
  probability_teachers_separated = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_teachers_separation_probability_l3355_335581


namespace NUMINAMATH_CALUDE_lakes_country_islands_l3355_335558

/-- A connected planar graph representing the lakes and canals system -/
structure LakeSystem where
  V : ℕ  -- number of vertices (lakes)
  E : ℕ  -- number of edges (canals)
  is_connected : Bool
  is_planar : Bool

/-- The number of islands in a lake system -/
def num_islands (sys : LakeSystem) : ℕ :=
  sys.V - sys.E + 2 - 1

/-- Theorem stating the number of islands in the given lake system -/
theorem lakes_country_islands (sys : LakeSystem) 
  (h1 : sys.V = 7)
  (h2 : sys.E = 10)
  (h3 : sys.is_connected = true)
  (h4 : sys.is_planar = true) :
  num_islands sys = 4 := by
  sorry

#eval num_islands ⟨7, 10, true, true⟩

end NUMINAMATH_CALUDE_lakes_country_islands_l3355_335558


namespace NUMINAMATH_CALUDE_julian_borrows_eight_l3355_335516

/-- The amount Julian borrows -/
def additional_borrowed (current_debt new_debt : ℕ) : ℕ :=
  new_debt - current_debt

/-- Proof that Julian borrows 8 dollars -/
theorem julian_borrows_eight :
  let current_debt := 20
  let new_debt := 28
  additional_borrowed current_debt new_debt = 8 := by
  sorry

end NUMINAMATH_CALUDE_julian_borrows_eight_l3355_335516


namespace NUMINAMATH_CALUDE_work_completion_time_l3355_335548

/-- 
Given:
- Person A can complete a work in 30 days
- Person A and B together complete 0.38888888888888884 part of the work in 7 days

Prove:
Person B can complete the work alone in 45 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 7 * (1 / a + 1 / b) = 0.38888888888888884) : b = 45 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3355_335548


namespace NUMINAMATH_CALUDE_price_per_working_game_l3355_335547

def total_games : ℕ := 10
def non_working_games : ℕ := 2
def total_earnings : ℕ := 32

theorem price_per_working_game :
  (total_earnings : ℚ) / (total_games - non_working_games) = 4 := by
  sorry

end NUMINAMATH_CALUDE_price_per_working_game_l3355_335547


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3355_335550

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l : Line)
  (A : Point)
  (given_line : Line)
  (h1 : A.liesOn l)
  (h2 : l.perpendicular given_line)
  (h3 : A.x = -1)
  (h4 : A.y = 3)
  (h5 : given_line.a = 1)
  (h6 : given_line.b = -2)
  (h7 : given_line.c = -3) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3355_335550


namespace NUMINAMATH_CALUDE_total_snow_is_0_53_l3355_335570

/-- The amount of snow on Monday in inches -/
def snow_monday : ℝ := 0.32

/-- The amount of snow on Tuesday in inches -/
def snow_tuesday : ℝ := 0.21

/-- The total amount of snow on Monday and Tuesday combined -/
def total_snow : ℝ := snow_monday + snow_tuesday

/-- Theorem stating that the total snow on Monday and Tuesday is 0.53 inches -/
theorem total_snow_is_0_53 : total_snow = 0.53 := by
  sorry

end NUMINAMATH_CALUDE_total_snow_is_0_53_l3355_335570


namespace NUMINAMATH_CALUDE_number_solution_l3355_335555

theorem number_solution : 
  ∀ (number : ℝ), (number * (-8) = 1600) → number = -200 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3355_335555


namespace NUMINAMATH_CALUDE_passenger_difference_l3355_335572

structure BusRoute where
  initial_passengers : ℕ
  first_passengers : ℕ
  final_passengers : ℕ
  terminal_passengers : ℕ

def BusRoute.valid (route : BusRoute) : Prop :=
  route.initial_passengers = 30 ∧
  route.terminal_passengers = 14 ∧
  route.first_passengers * 3 = route.final_passengers

theorem passenger_difference (route : BusRoute) (h : route.valid) :
  ∃ y : ℕ, route.first_passengers + y = route.initial_passengers + 6 :=
by sorry

end NUMINAMATH_CALUDE_passenger_difference_l3355_335572


namespace NUMINAMATH_CALUDE_lieutenant_age_l3355_335557

theorem lieutenant_age : ∃ (n : ℕ) (x : ℕ),
  n * (n + 5) = x * (n + 9) ∧
  x = 24 := by
  sorry

end NUMINAMATH_CALUDE_lieutenant_age_l3355_335557


namespace NUMINAMATH_CALUDE_license_plate_count_l3355_335580

def even_digits : Nat := 5
def consonants : Nat := 20
def vowels : Nat := 6

def license_plate_combinations : Nat :=
  even_digits * consonants * vowels * consonants

theorem license_plate_count :
  license_plate_combinations = 12000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3355_335580


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3355_335545

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3355_335545


namespace NUMINAMATH_CALUDE_pencil_weight_l3355_335521

theorem pencil_weight (total_weight : ℝ) (case_weight : ℝ) (num_pencils : ℕ) 
  (h1 : total_weight = 11.14)
  (h2 : case_weight = 0.5)
  (h3 : num_pencils = 14) :
  (total_weight - case_weight) / num_pencils = 0.76 := by
sorry

end NUMINAMATH_CALUDE_pencil_weight_l3355_335521


namespace NUMINAMATH_CALUDE_jeffreys_poultry_farm_l3355_335564

/-- The number of roosters for every 3 hens on Jeffrey's poultry farm -/
def roosters_per_three_hens : ℕ := by sorry

theorem jeffreys_poultry_farm :
  let total_hens : ℕ := 12
  let chicks_per_hen : ℕ := 5
  let total_chickens : ℕ := 76
  roosters_per_three_hens = 1 := by sorry

end NUMINAMATH_CALUDE_jeffreys_poultry_farm_l3355_335564


namespace NUMINAMATH_CALUDE_largest_area_is_16_l3355_335559

/-- Represents a polygon made of squares and right triangles -/
structure Polygon where
  num_squares : Nat
  num_triangles : Nat

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ :=
  4 * p.num_squares + 2 * p.num_triangles

/-- The set of all possible polygons in our problem -/
def polygon_set : Set Polygon :=
  { p | p.num_squares + p.num_triangles ≤ 4 }

theorem largest_area_is_16 :
  ∃ (p : Polygon), p ∈ polygon_set ∧ area p = 16 ∧ ∀ (q : Polygon), q ∈ polygon_set → area q ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_largest_area_is_16_l3355_335559


namespace NUMINAMATH_CALUDE_arc_length_from_sector_area_l3355_335586

/-- Given a circle with radius 5 cm and a sector with area 13.75 cm²,
    prove that the length of the arc forming the sector is 5.5 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) :
  r = 5 →
  area = 13.75 →
  arc_length = (2 * area) / r →
  arc_length = 5.5 :=
by
  sorry

#check arc_length_from_sector_area

end NUMINAMATH_CALUDE_arc_length_from_sector_area_l3355_335586


namespace NUMINAMATH_CALUDE_y_coordinate_is_1000_l3355_335594

/-- A straight line in the xy-plane with given properties -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point on a line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-coordinate of a point on a line can be calculated using the line's equation -/
def y_coordinate (l : Line) (p : Point) : ℝ :=
  l.slope * p.x + l.y_intercept

/-- Theorem: The y-coordinate of the specified point on the given line is 1000 -/
theorem y_coordinate_is_1000 (l : Line) (p : Point)
  (h1 : l.slope = 9.9)
  (h2 : l.y_intercept = 10)
  (h3 : p.x = 100) :
  y_coordinate l p = 1000 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_is_1000_l3355_335594


namespace NUMINAMATH_CALUDE_inequality_properties_l3355_335510

theorem inequality_properties (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a > c / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3355_335510


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3355_335596

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (1 - 3 * Complex.I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3355_335596


namespace NUMINAMATH_CALUDE_hamburger_buns_cost_l3355_335573

/-- The cost of hamburger buns given Lauren's grocery purchase --/
theorem hamburger_buns_cost : 
  ∀ (meat_price meat_weight lettuce_price tomato_price tomato_weight
     pickle_price pickle_discount paid change bun_price : ℝ),
  meat_price = 3.5 →
  meat_weight = 2 →
  lettuce_price = 1 →
  tomato_price = 2 →
  tomato_weight = 1.5 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  paid = 20 →
  change = 6 →
  bun_price = paid - change - (meat_price * meat_weight + lettuce_price + 
    tomato_price * tomato_weight + pickle_price - pickle_discount) →
  bun_price = 1.5 := by
sorry

end NUMINAMATH_CALUDE_hamburger_buns_cost_l3355_335573


namespace NUMINAMATH_CALUDE_area_of_ABCM_l3355_335585

/-- A 12-sided polygon with specific properties -/
structure TwelveSidedPolygon where
  /-- The length of each side of the polygon -/
  side_length : ℝ
  /-- The property that each two consecutive sides form a right angle -/
  right_angles : Bool

/-- The intersection point of two diagonals in the polygon -/
def IntersectionPoint (p : TwelveSidedPolygon) := Unit

/-- A quadrilateral formed by three vertices of the polygon and the intersection point -/
def Quadrilateral (p : TwelveSidedPolygon) (m : IntersectionPoint p) := Unit

/-- The area of a quadrilateral -/
def area (q : Quadrilateral p m) : ℝ := sorry

/-- Theorem stating the area of quadrilateral ABCM in the given polygon -/
theorem area_of_ABCM (p : TwelveSidedPolygon) (m : IntersectionPoint p) 
  (q : Quadrilateral p m) (h1 : p.side_length = 4) (h2 : p.right_angles = true) : 
  area q = 88 / 5 := by sorry

end NUMINAMATH_CALUDE_area_of_ABCM_l3355_335585


namespace NUMINAMATH_CALUDE_sequence_length_l3355_335578

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  n > 0 ∧
  b 0 = 41 ∧
  b 1 = 68 ∧
  b n = 0 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 698 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l3355_335578


namespace NUMINAMATH_CALUDE_seating_theorem_l3355_335568

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (abc : ℕ) (de : ℕ) : ℕ :=
  factorial n - (factorial (n - 2) * factorial 3) - 
  (factorial (n - 1) * factorial 2) + 
  (factorial (n - 3) * factorial 3 * factorial 2)

theorem seating_theorem : 
  seating_arrangements 10 3 2 = 2853600 := by sorry

end NUMINAMATH_CALUDE_seating_theorem_l3355_335568


namespace NUMINAMATH_CALUDE_steering_wheel_translational_on_straight_road_l3355_335513

/-- A road is considered straight if it has no curves or turns. -/
def is_straight_road (road : Type) : Prop := sorry

/-- A motion is translational if it involves no rotation. -/
def is_translational_motion (motion : Type) : Prop := sorry

/-- The steering wheel motion when driving on a given road. -/
def steering_wheel_motion (road : Type) : Type := sorry

/-- Theorem: The steering wheel motion is translational when driving on a straight road. -/
theorem steering_wheel_translational_on_straight_road (road : Type) :
  is_straight_road road → is_translational_motion (steering_wheel_motion road) := by sorry

end NUMINAMATH_CALUDE_steering_wheel_translational_on_straight_road_l3355_335513


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3355_335588

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_problem (h : ∀ y, y = f 2 → y = 2 + 4) : 
  f 2 + deriv f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l3355_335588


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l3355_335592

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 →
  Nat.gcd a.val b.val = 1 →
  a.val + b.val = 15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l3355_335592


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3355_335577

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = √3 x and one of its foci lies on the line x = -6,
    then the equation of the hyperbola is x²/9 - y²/27 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 6) →
  b/a = Real.sqrt 3 →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/27 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3355_335577


namespace NUMINAMATH_CALUDE_eyes_saw_airplane_l3355_335507

/-- Given 200 students and 3/4 of them looking up, prove that 300 eyes saw the airplane. -/
theorem eyes_saw_airplane (total_students : ℕ) (fraction_looked_up : ℚ) (h1 : total_students = 200) (h2 : fraction_looked_up = 3/4) :
  (fraction_looked_up * total_students : ℚ).num * 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_eyes_saw_airplane_l3355_335507


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3355_335504

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - Real.log x - a*x

theorem min_value_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f x 1 ≥ min := by sorry

theorem f_greater_than_x_iff_a_negative :
  (∀ x > 0, f x a > x) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3355_335504


namespace NUMINAMATH_CALUDE_work_completion_proof_l3355_335522

/-- Represents the time taken to complete the work -/
def total_days : ℕ := 11

/-- Represents the rate at which person a completes the work -/
def rate_a : ℚ := 1 / 24

/-- Represents the rate at which person b completes the work -/
def rate_b : ℚ := 1 / 30

/-- Represents the rate at which person c completes the work -/
def rate_c : ℚ := 1 / 40

/-- Represents the days c left before completion of work -/
def days_c_left : ℕ := 4

theorem work_completion_proof :
  ∃ (x : ℕ), x = days_c_left ∧
  (rate_a + rate_b + rate_c) * (total_days - x : ℚ) + (rate_a + rate_b) * x = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3355_335522


namespace NUMINAMATH_CALUDE_decimal_2_09_to_percentage_l3355_335517

/-- Converts a decimal number to a percentage -/
def decimal_to_percentage (x : ℝ) : ℝ := 100 * x

theorem decimal_2_09_to_percentage :
  decimal_to_percentage 2.09 = 209 := by sorry

end NUMINAMATH_CALUDE_decimal_2_09_to_percentage_l3355_335517


namespace NUMINAMATH_CALUDE_cylinder_volume_doubling_l3355_335554

/-- Given a cylinder with original volume 10 cubic feet, prove that doubling its height
    while keeping the radius constant results in a new volume of 20 cubic feet. -/
theorem cylinder_volume_doubling (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 10 → π * r^2 * (2 * h) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_doubling_l3355_335554


namespace NUMINAMATH_CALUDE_dance_club_average_age_l3355_335544

theorem dance_club_average_age 
  (num_females : Nat) 
  (num_males : Nat) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 12)
  (h2 : num_males = 18)
  (h3 : avg_age_females = 25)
  (h4 : avg_age_males = 40) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end NUMINAMATH_CALUDE_dance_club_average_age_l3355_335544


namespace NUMINAMATH_CALUDE_newsletter_cost_l3355_335509

def newsletter_cost_exists : Prop :=
  ∃ x : ℝ, 
    (14 * x < 16) ∧ 
    (19 * x > 21) ∧ 
    (∀ y : ℝ, (14 * y < 16) ∧ (19 * y > 21) → |x - 1.11| ≤ |y - 1.11|)

theorem newsletter_cost : newsletter_cost_exists := by sorry

end NUMINAMATH_CALUDE_newsletter_cost_l3355_335509


namespace NUMINAMATH_CALUDE_one_real_root_l3355_335552

-- Define the determinant function
def det (x c b d : ℝ) : ℝ := x^3 + (c^2 + d^2) * x

-- State the theorem
theorem one_real_root
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! x : ℝ, det x c b d = 0 :=
sorry

end NUMINAMATH_CALUDE_one_real_root_l3355_335552


namespace NUMINAMATH_CALUDE_inequality_system_solvability_l3355_335500

theorem inequality_system_solvability (n : ℕ) : 
  (∃ x : ℝ, 
    (1 < x ∧ x < 2) ∧
    (2 < x^2 ∧ x^2 < 3) ∧
    (∀ k : ℕ, 3 ≤ k ∧ k ≤ n → k < x^k ∧ x^k < k + 1)) ↔
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solvability_l3355_335500


namespace NUMINAMATH_CALUDE_local_extremum_cubic_l3355_335565

/-- Given a cubic function f(x) = ax³ + 3x² - 6ax + b with a local extremum of 9 at x = 2,
    prove that a + 2b = -24 -/
theorem local_extremum_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 - 6 * a * x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  f 2 = 9 →
  a + 2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_local_extremum_cubic_l3355_335565


namespace NUMINAMATH_CALUDE_find_A_l3355_335597

theorem find_A (A : ℕ) (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 999) 
  (h2 : 1000 * A + B = A * (A + 1) / 2) : A = 1999 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3355_335597


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_theorem_l3355_335546

-- Define a monic cubic polynomial with real coefficients
def monicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

-- State the theorem
theorem monic_cubic_polynomial_theorem (a b c : ℝ) :
  let q := monicCubicPolynomial a b c
  (q (3 - 2*I) = 0 ∧ q 0 = -108) →
  a = -(186/13) ∧ b = 1836/13 ∧ c = -108 :=
by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_theorem_l3355_335546


namespace NUMINAMATH_CALUDE_binomial_133_133_l3355_335524

theorem binomial_133_133 : Nat.choose 133 133 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_133_133_l3355_335524


namespace NUMINAMATH_CALUDE_acid_concentration_theorem_l3355_335567

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) : Prop :=
  let water1 := (acid1 / 0.05) - acid1
  let water2 := water - water1
  let conc2 := acid2 / (acid2 + water2)
  conc2 = 70 / 300 →
  let total_water := water1 + water2
  (acid3 / (acid3 + total_water)) * 100 = 10.5

theorem acid_concentration_theorem :
  acid_concentration_problem 10 20 30 255.714 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_theorem_l3355_335567


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3355_335529

theorem triangle_angle_not_all_greater_than_60 :
  ∀ (a b c : ℝ), 
  (a + b + c = 180) →  -- Sum of angles in a triangle is 180°
  (a > 0) → (b > 0) → (c > 0) →  -- All angles are positive
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3355_335529


namespace NUMINAMATH_CALUDE_fourth_square_dots_l3355_335514

/-- The number of dots in the nth square of the series -/
def dots_in_square (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_square (n - 1) + 4 * n

theorem fourth_square_dots :
  dots_in_square 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_dots_l3355_335514


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_135_point_5_l3355_335561

theorem x4_plus_y4_equals_135_point_5 (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 14) : 
  x^4 + y^4 = 135.5 := by
sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_135_point_5_l3355_335561


namespace NUMINAMATH_CALUDE_value_is_square_of_number_l3355_335576

theorem value_is_square_of_number (n v : ℕ) : 
  n = 14 → 
  v = n^2 → 
  n + v = 210 → 
  v = 196 := by sorry

end NUMINAMATH_CALUDE_value_is_square_of_number_l3355_335576


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3355_335520

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (2 - x) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3355_335520
