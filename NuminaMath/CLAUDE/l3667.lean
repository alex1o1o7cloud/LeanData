import Mathlib

namespace rectangle_max_area_l3667_366787

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
sorry

end rectangle_max_area_l3667_366787


namespace only_1680_is_product_of_four_consecutive_l3667_366793

/-- Given a natural number n, returns true if n can be expressed as a product of four consecutive natural numbers. -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) * (k + 2) * (k + 3)

/-- Theorem stating that among 712, 1262, and 1680, only 1680 can be expressed as a product of four consecutive natural numbers. -/
theorem only_1680_is_product_of_four_consecutive :
  ¬ is_product_of_four_consecutive 712 ∧
  ¬ is_product_of_four_consecutive 1262 ∧
  is_product_of_four_consecutive 1680 :=
by sorry

end only_1680_is_product_of_four_consecutive_l3667_366793


namespace basketball_players_count_l3667_366720

theorem basketball_players_count (total_athletes : ℕ) 
  (football_ratio baseball_ratio soccer_ratio basketball_ratio : ℕ) : 
  total_athletes = 104 →
  football_ratio = 10 →
  baseball_ratio = 7 →
  soccer_ratio = 5 →
  basketball_ratio = 4 →
  (basketball_ratio * total_athletes) / (football_ratio + baseball_ratio + soccer_ratio + basketball_ratio) = 16 := by
  sorry

end basketball_players_count_l3667_366720


namespace circular_garden_radius_l3667_366759

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2 → r = 12 := by
  sorry

end circular_garden_radius_l3667_366759


namespace commodity_price_problem_l3667_366774

theorem commodity_price_problem (price1 price2 : ℕ) : 
  price1 + price2 = 827 →
  price1 = price2 + 127 →
  price1 = 477 := by
sorry

end commodity_price_problem_l3667_366774


namespace max_sum_on_ellipse_l3667_366785

theorem max_sum_on_ellipse :
  ∀ x y : ℝ, (x - 2)^2 / 4 + (y - 1)^2 = 1 →
  ∀ x' y' : ℝ, (x' - 2)^2 / 4 + (y' - 1)^2 = 1 →
  x + y ≤ 3 + Real.sqrt 5 ∧
  ∃ x₀ y₀ : ℝ, (x₀ - 2)^2 / 4 + (y₀ - 1)^2 = 1 ∧ x₀ + y₀ = 3 + Real.sqrt 5 := by
  sorry

end max_sum_on_ellipse_l3667_366785


namespace z_reciprocal_modulus_l3667_366724

theorem z_reciprocal_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → 
  z = i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 → 
  Complex.abs (z⁻¹) = Real.sqrt 2 / 8 := by
  sorry

end z_reciprocal_modulus_l3667_366724


namespace bell_ringing_problem_l3667_366704

theorem bell_ringing_problem (S B : ℕ) : 
  S = (1/3 : ℚ) * B + 4 →
  B = 36 →
  S + B = 52 := by sorry

end bell_ringing_problem_l3667_366704


namespace hyperbola_chord_midpoint_l3667_366760

/-- Given a hyperbola x²/a² - y²/b² = 1 where a, b > 0,
    the midpoint of any chord with slope 1 lies on the line x/a² - y/b² = 0 -/
theorem hyperbola_chord_midpoint (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  x^2 / a^2 - y^2 / b^2 = 1 →
  ∃ (m : ℝ), (x + m) / a^2 - (y + m) / b^2 = 0 :=
by sorry

end hyperbola_chord_midpoint_l3667_366760


namespace unique_rectangle_with_given_perimeter_and_area_l3667_366725

theorem unique_rectangle_with_given_perimeter_and_area : 
  ∃! (w h : ℕ+), (2 * (w + h) = 80) ∧ (w * h = 400) :=
by sorry

end unique_rectangle_with_given_perimeter_and_area_l3667_366725


namespace regression_line_equation_l3667_366745

/-- Given a regression line with slope 1.23 passing through the point (4, 5),
    prove that its equation is ŷ = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (intercept : ℝ), 
    intercept = center_y - slope * center_x ∧
    intercept = 0.08 ∧
    ∀ (x y : ℝ), y = slope * x + intercept := by
  sorry

end regression_line_equation_l3667_366745


namespace initial_distance_between_trucks_l3667_366712

/-- Theorem: Initial distance between two trucks
Given:
- Two trucks X and Y traveling in the same direction
- Truck X's speed is 47 mph
- Truck Y's speed is 53 mph
- It takes 3 hours for Truck Y to overtake and be 5 miles ahead of Truck X
Prove: The initial distance between Truck X and Truck Y is 23 miles
-/
theorem initial_distance_between_trucks
  (speed_x : ℝ)
  (speed_y : ℝ)
  (overtake_time : ℝ)
  (ahead_distance : ℝ)
  (h1 : speed_x = 47)
  (h2 : speed_y = 53)
  (h3 : overtake_time = 3)
  (h4 : ahead_distance = 5)
  : ∃ (initial_distance : ℝ),
    initial_distance = (speed_y - speed_x) * overtake_time + ahead_distance :=
by
  sorry

end initial_distance_between_trucks_l3667_366712


namespace landscape_length_l3667_366737

/-- A rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playground_area : ℝ
  length_is_four_times_breadth : length = 4 * breadth
  playground_area_is_1200 : playground_area = 1200
  playground_is_one_third : playground_area = (1/3) * (length * breadth)

/-- The length of the landscape is 120 meters -/
theorem landscape_length (L : Landscape) : L.length = 120 := by
  sorry

end landscape_length_l3667_366737


namespace complex_imaginary_condition_l3667_366719

theorem complex_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 - 3*Complex.I) * (a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = 3 :=
by
  sorry

end complex_imaginary_condition_l3667_366719


namespace tiffany_lives_l3667_366767

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem: Given Tiffany's initial lives, lives lost, and lives gained,
    prove that her final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end tiffany_lives_l3667_366767


namespace one_tetrahedron_formed_l3667_366730

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the set of available triangles -/
def AvailableTriangles : Finset Triangle := sorry

/-- Checks if a set of four triangles can form a tetrahedron -/
def CanFormTetrahedron (t1 t2 t3 t4 : Triangle) : Prop := sorry

/-- Counts the number of tetrahedrons that can be formed -/
def CountTetrahedrons (triangles : Finset Triangle) : ℕ := sorry

/-- The main theorem stating that exactly one tetrahedron can be formed -/
theorem one_tetrahedron_formed :
  CountTetrahedrons AvailableTriangles = 1 := by sorry

end one_tetrahedron_formed_l3667_366730


namespace average_value_function_m_range_l3667_366794

/-- Definition of an average value function -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Ioo a b, f x₀ = (f b - f a) / (b - a)

/-- The function we're considering -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ x^2 - m*x - 1

/-- The theorem statement -/
theorem average_value_function_m_range :
  ∀ m : ℝ, is_average_value_function (f m) (-1) 1 → 0 < m ∧ m < 2 :=
sorry

end average_value_function_m_range_l3667_366794


namespace students_playing_neither_sport_l3667_366732

theorem students_playing_neither_sport (total : ℕ) (hockey : ℕ) (basketball : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_both : both = 10) :
  total - (hockey + basketball - both) = 4 := by
  sorry

end students_playing_neither_sport_l3667_366732


namespace power_of_product_l3667_366776

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end power_of_product_l3667_366776


namespace min_value_of_f_l3667_366773

noncomputable def f (x : ℝ) := x^2 + 2*x + 6/x + 9/x^2 + 4

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 10 + 4 * Real.sqrt 3 :=
by sorry

end min_value_of_f_l3667_366773


namespace tracy_art_fair_sales_l3667_366778

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (third_group : ℕ)
  (second_group_paintings : ℕ) (third_group_paintings : ℕ) (total_paintings_sold : ℕ)
  (h1 : total_customers = first_group + second_group + third_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 4)
  (h4 : second_group = 12)
  (h5 : third_group = 4)
  (h6 : second_group_paintings = 1)
  (h7 : third_group_paintings = 4)
  (h8 : total_paintings_sold = 36) :
  (total_paintings_sold - (second_group * second_group_paintings + third_group * third_group_paintings)) / first_group = 2 :=
sorry

end tracy_art_fair_sales_l3667_366778


namespace locus_is_circle_l3667_366721

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse (F₁ F₂ : ℝ × ℝ) where
  a : ℝ
  h : a > 0

/-- A point P on the ellipse -/
def PointOnEllipse (e : Ellipse F₁ F₂) (P : ℝ × ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * e.a

/-- The point Q extended from F₁P such that |PQ| = |PF₂| -/
def ExtendedPoint (P F₁ F₂ : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

/-- The theorem stating that the locus of Q is a circle -/
theorem locus_is_circle (F₁ F₂ : ℝ × ℝ) (e : Ellipse F₁ F₂) :
  ∀ P Q : ℝ × ℝ, PointOnEllipse e P → ExtendedPoint P F₁ F₂ Q →
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, dist Q center = radius :=
sorry

end locus_is_circle_l3667_366721


namespace distance_between_centers_l3667_366786

/-- Given a triangle with sides 5, 12, and 13, the distance between the centers
    of its inscribed and circumscribed circles is √65/2 -/
theorem distance_between_centers (a b c : ℝ) (h_sides : a = 5 ∧ b = 12 ∧ c = 13) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt ((circumradius - inradius) ^ 2 + (area / (a * b * c) * (a + b - c) * (b + c - a) * (c + a - b))) = Real.sqrt 65 / 2 :=
by sorry

end distance_between_centers_l3667_366786


namespace min_value_reciprocal_sum_l3667_366749

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 2) :
  1 / x + 2 / y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 2 ∧ 1 / x₀ + 2 / y₀ = 2 := by
  sorry

end min_value_reciprocal_sum_l3667_366749


namespace regression_lines_intersect_at_means_l3667_366726

/-- A linear regression line for a set of data points -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample means of a dataset -/
structure SampleMeans where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem stating that two regression lines for the same dataset intersect at the sample means -/
theorem regression_lines_intersect_at_means 
  (m n : RegressionLine) (means : SampleMeans) : 
  ∃ (x y : ℝ), 
    x = means.x_mean ∧ 
    y = means.y_mean ∧ 
    y = m.slope * x + m.intercept ∧ 
    y = n.slope * x + n.intercept := by
  sorry


end regression_lines_intersect_at_means_l3667_366726


namespace min_comparisons_correct_l3667_366739

/-- Represents a deck of cards numbered from 1 to n -/
def Deck (n : ℕ) := Fin n

/-- Checks if two numbers are consecutive -/
def are_consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a)

/-- The minimum number of comparisons needed to guarantee finding a consecutive pair -/
def min_comparisons (n : ℕ) := n - 2

theorem min_comparisons_correct (n : ℕ) (h : n ≥ 100) :
  ∀ (d : Deck n), 
    ∃ (f : Fin (min_comparisons n) → Deck n × Deck n),
      ∀ (g : Deck n × Deck n → Bool),
        (∀ (i j : Deck n), g (i, j) = true ↔ are_consecutive i.val j.val) →
        ∃ (i : Fin (min_comparisons n)), g (f i) = true :=
sorry

#check min_comparisons_correct

end min_comparisons_correct_l3667_366739


namespace ceiling_abs_negative_l3667_366771

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end ceiling_abs_negative_l3667_366771


namespace selling_price_calculation_l3667_366781

def calculate_selling_price (initial_price maintenance_cost repair_cost transportation_cost : ℝ)
  (tax_rate currency_loss_rate depreciation_rate profit_margin : ℝ) : ℝ :=
  let total_expenses := initial_price + maintenance_cost + repair_cost + transportation_cost
  let after_tax := total_expenses * (1 + tax_rate)
  let after_currency_loss := after_tax * (1 - currency_loss_rate)
  let after_depreciation := after_currency_loss * (1 - depreciation_rate)
  after_depreciation * (1 + profit_margin)

theorem selling_price_calculation :
  calculate_selling_price 10000 2000 5000 1000 0.1 0.05 0.15 0.5 = 23982.75 :=
by sorry

end selling_price_calculation_l3667_366781


namespace line_direction_vector_value_l3667_366779

def point := ℝ × ℝ

def direction_vector (a : ℝ) : point := (a, -2)

def line_passes_through (p1 p2 : point) (v : point) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * v.1, p1.2 + t * v.2)

theorem line_direction_vector_value :
  ∀ a : ℝ,
  line_passes_through (-3, 6) (2, -1) (direction_vector a) →
  a = 10/7 := by
sorry

end line_direction_vector_value_l3667_366779


namespace unique_integer_solution_l3667_366702

theorem unique_integer_solution : ∃! (n : ℤ), n + 10 > 11 ∧ -4*n > -12 := by
  sorry

end unique_integer_solution_l3667_366702


namespace fred_initial_cards_l3667_366700

/-- The number of baseball cards Keith bought from Fred -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has now -/
def cards_remaining : ℕ := 18

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := cards_bought + cards_remaining

theorem fred_initial_cards : initial_cards = 40 := by
  sorry

end fred_initial_cards_l3667_366700


namespace specific_pairing_probability_l3667_366713

/-- The probability of a specific pairing in a class with random pairings. -/
theorem specific_pairing_probability
  (total_students : ℕ)
  (non_participating : ℕ)
  (h1 : total_students = 32)
  (h2 : non_participating = 1)
  : (1 : ℚ) / (total_students - non_participating - 1) = 1 / 30 :=
by sorry

end specific_pairing_probability_l3667_366713


namespace dance_attendance_l3667_366764

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end dance_attendance_l3667_366764


namespace like_terms_imply_sum_l3667_366756

/-- Two terms are like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c, term1 x y = c * term2 x y

/-- The first term in our problem -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^(2*m) * y^m

/-- The second term in our problem -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := x^(4-n) * y^(n-1)

theorem like_terms_imply_sum (m n : ℕ) : 
  like_terms (term1 m) (term2 n) → m + n = 3 := by
sorry

end like_terms_imply_sum_l3667_366756


namespace storks_and_birds_l3667_366701

theorem storks_and_birds (initial_storks initial_birds new_birds : ℕ) :
  initial_storks = 6 →
  initial_birds = 2 →
  new_birds = 3 →
  initial_storks - (initial_birds + new_birds) = 1 :=
by
  sorry

end storks_and_birds_l3667_366701


namespace combined_liquid_fraction_l3667_366729

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℝ
  filled : ℝ
  density : ℝ

/-- The problem setup -/
def problemSetup : Prop := ∃ (small large third : Beaker),
  -- Small beaker conditions
  small.filled = (1/2) * small.capacity ∧
  small.density = 1.025 ∧
  -- Large beaker conditions
  large.capacity = 5 * small.capacity ∧
  large.filled = (1/5) * large.capacity ∧
  large.density = 1 ∧
  -- Third beaker conditions
  third.capacity = (1/2) * large.capacity ∧
  third.filled = (3/4) * third.capacity ∧
  third.density = 0.85

/-- The theorem to prove -/
theorem combined_liquid_fraction (h : problemSetup) :
  ∃ (small large third : Beaker),
  (large.filled + small.filled + third.filled) / large.capacity = 27/40 := by
  sorry

end combined_liquid_fraction_l3667_366729


namespace road_repair_hours_l3667_366780

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 33)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 11)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 
        people2 * days2 * hours2 * (people1 * days1).lcm (people2 * days2 * hours2) / (people2 * days2 * hours2)) :
  (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 5 := by
  sorry

end road_repair_hours_l3667_366780


namespace arithmetic_sequence_problem_l3667_366789

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3)
  (h2 : a 11 = 15)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 1 = 0 ∧ a 2 - a 1 = 3/2 :=
by sorry

end arithmetic_sequence_problem_l3667_366789


namespace negation_equivalence_l3667_366735

/-- A function f: ℝ → ℝ is monotonically increasing on (0, +∞) if for all x₁, x₂ ∈ (0, +∞),
    x₁ < x₂ implies f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The negation of the existence of a real k such that y = k/x is monotonically increasing
    on (0, +∞) is equivalent to the statement that for all real k, y = k/x is not
    monotonically increasing on (0, +∞) -/
theorem negation_equivalence : 
  (¬ ∃ k : ℝ, MonotonicallyIncreasing (fun x ↦ k / x)) ↔ 
  (∀ k : ℝ, ¬ MonotonicallyIncreasing (fun x ↦ k / x)) := by
  sorry

end negation_equivalence_l3667_366735


namespace insufficient_comparisons_l3667_366716

/-- Represents a comparison of three elements -/
structure TripleComparison (α : Type) where
  a : α
  b : α
  c : α

/-- The type of all possible orderings of n distinct elements -/
def Orderings (n : ℕ) := Fin n → Fin n

/-- The number of possible orderings for n distinct elements -/
def num_orderings (n : ℕ) : ℕ := n.factorial

/-- The maximum number of orderings that can be eliminated by a single triple comparison -/
def max_eliminated_by_comparison (n : ℕ) : ℕ := (n - 2).factorial

/-- The number of comparisons allowed -/
def num_comparisons : ℕ := 9

/-- The number of distinct elements to be ordered -/
def num_elements : ℕ := 5

/-- Theorem stating that the given number of comparisons is insufficient -/
theorem insufficient_comparisons :
  ∃ (remaining : ℕ), remaining > 1 ∧
  remaining ≤ num_orderings num_elements - num_comparisons * max_eliminated_by_comparison num_elements :=
sorry

end insufficient_comparisons_l3667_366716


namespace triangle_sine_problem_l3667_366740

theorem triangle_sine_problem (D E F : ℝ) (h_area : (1/2) * D * E * Real.sin F = 72) 
  (h_geometric_mean : Real.sqrt (D * E) = 15) : Real.sin F = 16/25 := by
  sorry

end triangle_sine_problem_l3667_366740


namespace defective_tubes_count_l3667_366748

/-- The probability of selecting two defective tubes without replacement -/
def prob_two_defective : ℝ := 0.05263157894736842

/-- The total number of picture tubes in the consignment -/
def total_tubes : ℕ := 20

/-- The number of defective picture tubes in the consignment -/
def num_defective : ℕ := 5

theorem defective_tubes_count :
  (num_defective : ℝ) / total_tubes * ((num_defective - 1) : ℝ) / (total_tubes - 1) = prob_two_defective := by
  sorry

end defective_tubes_count_l3667_366748


namespace sqrt_five_position_l3667_366710

/-- Given a sequence where the square of the n-th term is 3n - 1, 
    prove that 2√5 is the 7th term of this sequence. -/
theorem sqrt_five_position (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ n, a n ^ 2 = 3 * n - 1) : 
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end sqrt_five_position_l3667_366710


namespace ammeter_readings_sum_l3667_366743

/-- The sum of readings of five ammeters in a specific circuit configuration -/
def sum_of_ammeter_readings (I₁ I₂ I₃ I₄ I₅ : ℝ) : ℝ :=
  I₁ + I₂ + I₃ + I₄ + I₅

/-- Theorem stating the sum of ammeter readings in the given circuit -/
theorem ammeter_readings_sum :
  ∀ (I₁ I₂ I₃ I₄ I₅ : ℝ),
    I₁ = 2 →
    I₂ = I₁ →
    I₃ = I₁ + I₂ →
    I₅ = I₃ + I₁ →
    I₄ = (5/3) * I₅ →
    sum_of_ammeter_readings I₁ I₂ I₃ I₄ I₅ = 24 := by
  sorry


end ammeter_readings_sum_l3667_366743


namespace beach_population_l3667_366755

theorem beach_population (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
sorry

end beach_population_l3667_366755


namespace additional_employees_hired_l3667_366796

/-- Calculates the number of additional employees hired by a company --/
theorem additional_employees_hired (
  initial_employees : ℕ)
  (hourly_wage : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (new_total_wages : ℚ)
  (h1 : initial_employees = 500)
  (h2 : hourly_wage = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : new_total_wages = 1680000) :
  (new_total_wages - (initial_employees * hourly_wage * hours_per_day * days_per_week * weeks_per_month)) / 
  (hourly_wage * hours_per_day * days_per_week * weeks_per_month) = 200 := by
  sorry

#check additional_employees_hired

end additional_employees_hired_l3667_366796


namespace sqrt_x_minus_one_real_l3667_366753

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l3667_366753


namespace total_seashells_eq_sum_l3667_366792

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Joan gave to Mike -/
def seashells_given : ℕ := 63

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 16

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_eq_sum : total_seashells = seashells_given + seashells_left := by sorry

end total_seashells_eq_sum_l3667_366792


namespace range_equivalence_l3667_366768

/-- The function f(x) = -x³ + 3bx --/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + 3*b*x

/-- The theorem stating the equivalence between the range of f and the value of b --/
theorem range_equivalence (b : ℝ) :
  (∀ y ∈ Set.range (f b), y ∈ Set.Icc 0 1) ∧
  (∀ y ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc 0 1, f b x = y) ↔
  b = (2 : ℝ)^(1/3) / 2 := by
sorry

end range_equivalence_l3667_366768


namespace sufficient_but_not_necessary_l3667_366722

-- Define the sets
def set1 : Set ℝ := {x | x * (x - 3) < 0}
def set2 : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem sufficient_but_not_necessary : set1 ⊆ set2 ∧ set1 ≠ set2 := by
  sorry

end sufficient_but_not_necessary_l3667_366722


namespace triangle_properties_l3667_366718

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c →
  Real.sin A * Real.sin B = (Real.cos (C / 2))^2 →
  ((a^2 + b^2 - c^2) / 4 + (c^2 * (Real.cos (C / 2))^2)) = 7 →
  A = Real.pi / 6 ∧ B = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end triangle_properties_l3667_366718


namespace stratified_sampling_male_athletes_l3667_366777

/-- Represents the number of male athletes in a stratified sample -/
def male_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (male_athletes * sample_size) / total_athletes

/-- Theorem: In a stratified sampling of 28 athletes from a team of 98 athletes (56 male and 42 female),
    the number of male athletes in the sample should be 16. -/
theorem stratified_sampling_male_athletes :
  male_athletes_in_sample 98 56 28 = 16 := by
  sorry

end stratified_sampling_male_athletes_l3667_366777


namespace arithmetic_sequence_min_value_b_l3667_366769

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def cosine_condition (t : Triangle) : Prop :=
  2 * Real.cos t.B * (t.c * Real.cos t.A + t.a * Real.cos t.C) = t.b

def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2

-- Theorem 1: Arithmetic sequence
theorem arithmetic_sequence (t : Triangle) 
  (h1 : triangle_condition t) (h2 : cosine_condition t) : 
  ∃ r : Real, t.A = t.B - r ∧ t.C = t.B + r :=
sorry

-- Theorem 2: Minimum value of b
theorem min_value_b (t : Triangle) 
  (h1 : triangle_condition t) (h2 : area_condition t) : 
  t.b ≥ Real.sqrt 6 :=
sorry

end arithmetic_sequence_min_value_b_l3667_366769


namespace childrens_home_total_l3667_366758

theorem childrens_home_total (toddlers teenagers newborns : ℕ) : 
  teenagers = 5 * toddlers →
  toddlers = 6 →
  newborns = 4 →
  toddlers + teenagers + newborns = 40 :=
by
  sorry

end childrens_home_total_l3667_366758


namespace fidos_yard_area_l3667_366798

theorem fidos_yard_area (s : ℝ) (h : s > 0) :
  let r := s / 2
  let area_circle := π * r^2
  let area_square := s^2
  let fraction := area_circle / area_square
  ∃ (a b : ℝ), fraction = (Real.sqrt a / b) * π ∧ a * b = 0 :=
by sorry

end fidos_yard_area_l3667_366798


namespace exact_three_primes_l3667_366784

/-- The polynomial function f(n) = n^3 - 8n^2 + 20n - 13 -/
def f (n : ℕ) : ℤ := n^3 - 8*n^2 + 20*n - 13

/-- Predicate for primality -/
def isPrime (n : ℤ) : Prop := n > 1 ∧ (∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0))

theorem exact_three_primes : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n ∈ s, isPrime (f n) ∧ 
    ∀ n : ℕ, n > 0 → isPrime (f n) → n ∈ s :=
sorry

end exact_three_primes_l3667_366784


namespace HNO3_calculation_l3667_366738

-- Define the chemical equation
def chemical_equation : String := "CaO + 2 HNO₃ → Ca(NO₃)₂ + H₂O"

-- Define the initial amount of CaO in moles
def initial_CaO : ℝ := 7

-- Define the stoichiometric ratio of HNO₃ to CaO
def stoichiometric_ratio : ℝ := 2

-- Define atomic weights
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Theorem to prove
theorem HNO3_calculation (chemical_equation : String) (initial_CaO : ℝ) 
  (stoichiometric_ratio : ℝ) (atomic_weight_H : ℝ) (atomic_weight_N : ℝ) 
  (atomic_weight_O : ℝ) :
  let moles_HNO3 : ℝ := initial_CaO * stoichiometric_ratio
  let molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O
  (moles_HNO3 = 14 ∧ molecular_weight_HNO3 = 63.02) :=
by
  sorry

end HNO3_calculation_l3667_366738


namespace number_times_fifteen_equals_150_l3667_366709

theorem number_times_fifteen_equals_150 :
  ∃ x : ℝ, 15 * x = 150 ∧ x = 10 := by
  sorry

end number_times_fifteen_equals_150_l3667_366709


namespace cubic_function_derivative_l3667_366751

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b = f'(2), 
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_function_derivative (a b : ℝ) : 
  let f := fun x : ℝ => a * x^3 + b * x^2 + 3
  let f' := fun x : ℝ => 3 * a * x^2 + 2 * b * x
  (f' 1 = -5 ∧ b = f' 2) → f' 2 = -4 := by
  sorry

end cubic_function_derivative_l3667_366751


namespace inequalities_theorem_l3667_366788

theorem inequalities_theorem :
  (∀ (a b c d : ℝ), a > b → c > d → a - d > b - c) ∧
  (∀ (a b : ℝ), 1/a < 1/b → 1/b < 0 → a*b < b^2) :=
by sorry

end inequalities_theorem_l3667_366788


namespace max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l3667_366791

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := C p.1 p.2

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop := p.1 = -q.1 ∧ p.2 = -q.2

-- Statement 1: Maximum distance between symmetric points
theorem max_distance_symmetric_points :
  ∀ A B : ℝ × ℝ, on_ellipse A → on_ellipse B → symmetric_wrt_origin A B →
  ‖A - B‖ ≤ 10 :=
sorry

-- Statement 2: Constant sum of distances to foci
theorem constant_sum_distances_to_foci :
  ∀ A : ℝ × ℝ, on_ellipse A →
  ‖A - F₁‖ + ‖A - F₂‖ = 10 :=
sorry

-- Statement 3: Ratio of focal length to minor axis
theorem focal_length_to_minor_axis_ratio :
  ‖F₁ - F₂‖ / 8 = 3/4 :=
sorry

-- Statement 4: No point with perpendicular lines to foci
theorem no_perpendicular_lines_to_foci :
  ¬ ∃ A : ℝ × ℝ, on_ellipse A ∧ 
  (A - F₁) • (A - F₂) = 0 :=
sorry

end max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l3667_366791


namespace gain_percent_calculation_l3667_366708

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 25) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end gain_percent_calculation_l3667_366708


namespace parabola_properties_l3667_366763

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 6*x - 1

-- Theorem statement
theorem parabola_properties :
  -- Vertex coordinates
  (∃ (x y : ℝ), x = -3 ∧ y = -10 ∧ ∀ (t : ℝ), f t ≥ f x) ∧
  -- Axis of symmetry
  (∀ (x : ℝ), f (x - 3) = f (-x - 3)) ∧
  -- Y-axis intersection point
  f 0 = -1 := by
  sorry

end parabola_properties_l3667_366763


namespace symmetry_example_l3667_366754

/-- A point in 3D space is represented by its x, y, and z coordinates. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are the same,
    and their y and z coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = -q.y ∧ p.z = -q.z

/-- The theorem states that the point (-2, -1, -4) is symmetric to the point (-2, 1, 4)
    with respect to the x-axis. -/
theorem symmetry_example : 
  symmetric_wrt_x_axis (Point3D.mk (-2) 1 4) (Point3D.mk (-2) (-1) (-4)) := by
  sorry

end symmetry_example_l3667_366754


namespace bird_count_theorem_l3667_366761

/-- The number of birds on a fence after a series of additions and removals --/
def final_bird_count (initial : ℕ) (first_add : ℕ) (first_remove : ℕ) (second_add : ℕ) (third_add : ℚ) : ℚ :=
  let T : ℕ := initial + first_add
  let W : ℕ := T - first_remove + second_add
  (W : ℚ) / 2 + third_add

/-- Theorem stating the final number of birds on the fence --/
theorem bird_count_theorem : 
  final_bird_count 12 8 5 3 (5/2) = 23/2 := by sorry

end bird_count_theorem_l3667_366761


namespace fib_inequality_l3667_366799

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proof of the inequality for Fibonacci numbers -/
theorem fib_inequality (n : ℕ) (hn : n > 0) :
  (fib (n + 2) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fib (n + 1) : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end fib_inequality_l3667_366799


namespace rectangular_prism_volume_l3667_366733

theorem rectangular_prism_volume 
  (m n Q : ℝ) 
  (m_pos : m > 0) 
  (n_pos : n > 0) 
  (Q_pos : Q > 0) : 
  let base_ratio := m / n
  let diagonal_area := Q
  let volume := (m * n * Q * Real.sqrt Q) / (m^2 + n^2)
  ∃ (a b h : ℝ), 
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    a / b = base_ratio ∧
    a * a + b * b = Q ∧
    h * h = Q ∧
    a * b * h = volume :=
by sorry

end rectangular_prism_volume_l3667_366733


namespace proportional_function_m_value_l3667_366703

/-- A proportional function passing through a specific point -/
def proportional_function_through_point (k m : ℝ) : Prop :=
  4 * 2 = 3 - m

/-- Theorem: If the proportional function y = 4x passes through (2, 3-m), then m = -5 -/
theorem proportional_function_m_value (m : ℝ) :
  proportional_function_through_point 4 m → m = -5 := by
  sorry

end proportional_function_m_value_l3667_366703


namespace historical_fiction_new_releases_fraction_l3667_366714

/-- Represents the inventory composition and new release percentages for a bookstore. -/
structure BookstoreInventory where
  historicalFictionPercentage : Float
  scienceFictionPercentage : Float
  biographiesPercentage : Float
  mysteryNovelsPercentage : Float
  historicalFictionNewReleasePercentage : Float
  scienceFictionNewReleasePercentage : Float
  biographiesNewReleasePercentage : Float
  mysteryNovelsNewReleasePercentage : Float

/-- Calculates the fraction of all new releases that are historical fiction new releases. -/
def historicalFictionNewReleasesFraction (inventory : BookstoreInventory) : Float :=
  let totalNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage +
    inventory.scienceFictionPercentage * inventory.scienceFictionNewReleasePercentage +
    inventory.biographiesPercentage * inventory.biographiesNewReleasePercentage +
    inventory.mysteryNovelsPercentage * inventory.mysteryNovelsNewReleasePercentage
  let historicalFictionNewReleases := 
    inventory.historicalFictionPercentage * inventory.historicalFictionNewReleasePercentage
  historicalFictionNewReleases / totalNewReleases

/-- Theorem stating that the fraction of all new releases that are historical fiction new releases is 9/20. -/
theorem historical_fiction_new_releases_fraction :
  let inventory := BookstoreInventory.mk 0.40 0.25 0.15 0.20 0.45 0.30 0.50 0.35
  historicalFictionNewReleasesFraction inventory = 9/20 := by
  sorry

end historical_fiction_new_releases_fraction_l3667_366714


namespace find_y_value_l3667_366790

theorem find_y_value : ∃ y : ℝ, (15^2 * y^3) / 256 = 450 ∧ y = 8 := by
  sorry

end find_y_value_l3667_366790


namespace complex_arithmetic_l3667_366742

theorem complex_arithmetic (z : ℂ) (h : z = 1 + I) : (2 / z) + z^2 = 1 + I := by sorry

end complex_arithmetic_l3667_366742


namespace al2co3_3_weight_l3667_366736

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecular_weight (al_weight c_weight o_weight : ℝ) (num_moles : ℝ) : ℝ :=
  let co3_weight := c_weight + 3 * o_weight
  let al2co3_3_weight := 2 * al_weight + 3 * co3_weight
  num_moles * al2co3_3_weight

/-- Theorem stating the molecular weight of 6 moles of Al2(CO3)3 -/
theorem al2co3_3_weight : 
  molecular_weight 26.98 12.01 16.00 6 = 1403.94 := by
  sorry

end al2co3_3_weight_l3667_366736


namespace permutations_of_33377_l3667_366765

/-- The number of permutations of a multiset with 5 elements, where 3 elements are the same and 2 elements are the same -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

theorem permutations_of_33377 : permutations_of_multiset = 10 := by
  sorry

end permutations_of_33377_l3667_366765


namespace arithmetic_sequence_general_term_l3667_366707

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = -3) : 
  ∀ n : ℕ, a n = -2 * n + 3 := by
sorry

end arithmetic_sequence_general_term_l3667_366707


namespace number_of_girls_in_group_l3667_366746

theorem number_of_girls_in_group (girls_avg_weight : ℝ) (boys_avg_weight : ℝ) 
  (total_avg_weight : ℝ) (num_boys : ℕ) (total_students : ℕ) :
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  num_boys = 5 →
  total_students = 10 →
  total_avg_weight = 50 →
  ∃ (num_girls : ℕ), num_girls = 5 ∧ 
    (girls_avg_weight * num_girls + boys_avg_weight * num_boys) / total_students = total_avg_weight :=
by sorry

end number_of_girls_in_group_l3667_366746


namespace binomial_coefficient_sum_l3667_366795

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end binomial_coefficient_sum_l3667_366795


namespace complex_modulus_proof_l3667_366715

theorem complex_modulus_proof : 
  let z : ℂ := Complex.mk (3/4) (-5/6)
  ‖z‖ = Real.sqrt 127 / 12 := by sorry

end complex_modulus_proof_l3667_366715


namespace sqrt_y_fourth_power_l3667_366711

theorem sqrt_y_fourth_power (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end sqrt_y_fourth_power_l3667_366711


namespace y_equals_negative_two_at_x_two_l3667_366734

/-- A linear function y = kx - 1 where y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  h1 : k < 0

/-- The value of y when x = 2 for a decreasing linear function -/
def y_at_2 (f : DecreasingLinearFunction) : ℝ :=
  f.k * 2 - 1

/-- Theorem stating that y = -2 when x = 2 for a decreasing linear function -/
theorem y_equals_negative_two_at_x_two (f : DecreasingLinearFunction) :
  y_at_2 f = -2 :=
sorry

end y_equals_negative_two_at_x_two_l3667_366734


namespace inequality_solution_set_l3667_366750

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) ↔ |k * x - 4| ≤ 2) → k = 2 := by
  sorry

end inequality_solution_set_l3667_366750


namespace unknown_denomination_is_500_l3667_366797

/-- Represents the denomination problem with given conditions --/
structure DenominationProblem where
  total_amount : ℕ
  known_denomination : ℕ
  total_notes : ℕ
  known_denomination_count : ℕ
  (total_amount_check : total_amount = 10350)
  (known_denomination_check : known_denomination = 50)
  (total_notes_check : total_notes = 54)
  (known_denomination_count_check : known_denomination_count = 37)

/-- Theorem stating that the unknown denomination is 500 --/
theorem unknown_denomination_is_500 (p : DenominationProblem) : 
  (p.total_amount - p.known_denomination * p.known_denomination_count) / (p.total_notes - p.known_denomination_count) = 500 :=
sorry

end unknown_denomination_is_500_l3667_366797


namespace first_degree_function_characterization_l3667_366706

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem first_degree_function_characterization
  (f : ℝ → ℝ) 
  (h1 : FirstDegreeFunction f)
  (h2 : ∀ x : ℝ, f (f x) = 4 * x + 6) :
  (∀ x : ℝ, f x = 2 * x + 2) ∨ (∀ x : ℝ, f x = -2 * x - 6) :=
sorry

end first_degree_function_characterization_l3667_366706


namespace largest_touching_sphere_radius_l3667_366782

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  isRegular : Bool
  /-- The tetrahedron is inscribed in a unit sphere -/
  isInscribed : Bool

/-- A sphere touching the unit sphere internally and the tetrahedron externally -/
structure TouchingSphere where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The sphere touches the unit sphere internally -/
  touchesUnitSphereInternally : Bool
  /-- The sphere touches the tetrahedron externally -/
  touchesTetrahedronExternally : Bool

/-- The theorem stating the radius of the largest touching sphere -/
theorem largest_touching_sphere_radius 
  (t : InscribedTetrahedron) 
  (s : TouchingSphere) 
  (h1 : t.isRegular = true) 
  (h2 : t.isInscribed = true)
  (h3 : s.touchesUnitSphereInternally = true)
  (h4 : s.touchesTetrahedronExternally = true) :
  s.radius = 1/3 :=
sorry

end largest_touching_sphere_radius_l3667_366782


namespace count_four_digit_integers_thousands_4_l3667_366775

/-- The count of four-digit positive integers with the thousands digit 4 -/
def fourDigitIntegersWithThousands4 : ℕ :=
  (Finset.range 10).card * (Finset.range 10).card * (Finset.range 10).card

/-- Theorem stating that the count of four-digit positive integers with the thousands digit 4 is 1000 -/
theorem count_four_digit_integers_thousands_4 :
  fourDigitIntegersWithThousands4 = 1000 := by sorry

end count_four_digit_integers_thousands_4_l3667_366775


namespace square_area_increase_l3667_366723

/-- The increase in area of a square when its side length is increased -/
theorem square_area_increase (initial_side : ℝ) (increase : ℝ) : 
  initial_side = 6 → increase = 1 → 
  (initial_side + increase)^2 - initial_side^2 = 13 := by
  sorry

#check square_area_increase

end square_area_increase_l3667_366723


namespace max_value_of_expression_l3667_366783

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5) ≤ 17 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 5) = 17 := by
  sorry

end max_value_of_expression_l3667_366783


namespace value_of_2a_minus_3b_l3667_366717

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_3b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end value_of_2a_minus_3b_l3667_366717


namespace equation_always_has_two_solutions_l3667_366772

theorem equation_always_has_two_solutions (b : ℝ) (h : 1 ≤ b ∧ b ≤ 25) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^4 + 36*b^2 = (9*b^2 - 15*b)*x₁^2 ∧
  x₂^4 + 36*b^2 = (9*b^2 - 15*b)*x₂^2 :=
sorry

end equation_always_has_two_solutions_l3667_366772


namespace probability_theorem_l3667_366766

def num_events : ℕ := 5
def prob_success : ℚ := 3/4

theorem probability_theorem :
  (prob_success ^ num_events = 243/1024) ∧
  (1 - (1 - prob_success) ^ num_events = 1023/1024) :=
sorry

end probability_theorem_l3667_366766


namespace decimal_division_l3667_366770

theorem decimal_division (x y : ℚ) (hx : x = 0.45) (hy : y = 0.005) : x / y = 90 := by
  sorry

end decimal_division_l3667_366770


namespace xiaogang_dart_game_l3667_366752

theorem xiaogang_dart_game :
  ∀ (x y z : ℕ),
    x + y + z > 11 →
    8 * x + 9 * y + 10 * z = 100 →
    (x + y + z = 12 ∧ (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end xiaogang_dart_game_l3667_366752


namespace trajectory_is_line_with_equal_tangents_l3667_366741

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 - 4 = (x - 3)^2 + (y - 2)^2 - 1

-- Define tangent length squared to O1
def tangent_length_sq_O1 (x y : ℝ) : ℝ := (x + 1)^2 + (y + 1)^2 - 4

-- Define tangent length squared to O2
def tangent_length_sq_O2 (x y : ℝ) : ℝ := (x - 3)^2 + (y - 2)^2 - 1

-- Theorem statement
theorem trajectory_is_line_with_equal_tangents :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, trajectory x y ↔ a * x + b * y + c = 0) ∧
    (∀ x y : ℝ, trajectory x y → tangent_length_sq_O1 x y = tangent_length_sq_O2 x y) :=
sorry

end trajectory_is_line_with_equal_tangents_l3667_366741


namespace required_third_subject_score_l3667_366762

def average_score_two_subjects : ℝ := 88
def target_average_three_subjects : ℝ := 90
def number_of_subjects : ℕ := 3

theorem required_third_subject_score :
  let total_score_two_subjects := average_score_two_subjects * 2
  let total_score_three_subjects := target_average_three_subjects * number_of_subjects
  total_score_three_subjects - total_score_two_subjects = 94 := by
  sorry

end required_third_subject_score_l3667_366762


namespace gcd_of_42_77_105_l3667_366731

theorem gcd_of_42_77_105 : Nat.gcd 42 (Nat.gcd 77 105) = 7 := by sorry

end gcd_of_42_77_105_l3667_366731


namespace operation_on_number_l3667_366757

theorem operation_on_number (x : ℝ) : x^2 = 25 → 2*x = x/5 + 9 := by
  sorry

end operation_on_number_l3667_366757


namespace otimes_nested_equal_101_l3667_366744

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := b^2 + 1

-- Theorem statement
theorem otimes_nested_equal_101 (m : ℚ) : otimes m (otimes m 3) = 101 := by
  sorry

end otimes_nested_equal_101_l3667_366744


namespace simon_beach_treasures_l3667_366728

/-- Represents the number of treasures Simon collected on the beach. -/
def beach_treasures (sand_dollars : ℕ) (glass_multiplier : ℕ) (shell_multiplier : ℕ) : ℕ :=
  let glass := sand_dollars * glass_multiplier
  let shells := glass * shell_multiplier
  sand_dollars + glass + shells

/-- Proves that Simon collected 190 treasures on the beach. -/
theorem simon_beach_treasures :
  beach_treasures 10 3 5 = 190 := by
  sorry

end simon_beach_treasures_l3667_366728


namespace expression_value_l3667_366747

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by
  sorry

end expression_value_l3667_366747


namespace min_value_z_l3667_366705

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := 4*x + y

/-- The feasible region defined by the constraints -/
def feasible_region (x y : ℝ) : Prop :=
  3*x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

/-- The theorem stating that the minimum value of z in the feasible region is 7 -/
theorem min_value_z :
  ∀ x y : ℝ, feasible_region x y → z x y ≥ 7 ∧ ∃ x₀ y₀ : ℝ, feasible_region x₀ y₀ ∧ z x₀ y₀ = 7 :=
sorry

end min_value_z_l3667_366705


namespace wholesale_price_calculation_l3667_366727

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup factor applied to the wholesale price -/
def markup_factor : ℝ := 1.8

theorem wholesale_price_calculation :
  wholesale_price * markup_factor = retail_price :=
by sorry

end wholesale_price_calculation_l3667_366727
