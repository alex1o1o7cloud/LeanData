import Mathlib

namespace absolute_value_zero_l2343_234385

theorem absolute_value_zero (y : ℚ) : |2 * y - 3| = 0 ↔ y = 3/2 := by sorry

end absolute_value_zero_l2343_234385


namespace total_legs_calculation_l2343_234393

/-- The total number of legs of Camden's dogs, Rico's dogs, and Samantha's cats -/
def totalLegs : ℕ := by sorry

theorem total_legs_calculation :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dogs : ℕ := (3 * rico_dogs) / 4
  let camden_legs : ℕ := 5 * 3 + 7 * 4 + 2 * 2
  let rico_legs : ℕ := rico_dogs * 4
  let samantha_cats : ℕ := 8
  let samantha_legs : ℕ := 6 * 4 + 2 * 3
  totalLegs = camden_legs + rico_legs + samantha_legs := by sorry

end total_legs_calculation_l2343_234393


namespace A_intersect_B_eq_singleton_one_l2343_234377

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem A_intersect_B_eq_singleton_one : A ∩ B = {1} := by
  sorry

end A_intersect_B_eq_singleton_one_l2343_234377


namespace a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l2343_234360

theorem a_neq_1_necessary_not_sufficient_for_a_squared_neq_1 :
  (∀ a : ℝ, a^2 ≠ 1 → a ≠ 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ a^2 = 1) :=
by sorry

end a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l2343_234360


namespace x_squared_minus_y_squared_l2343_234366

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end x_squared_minus_y_squared_l2343_234366


namespace vector_sum_example_l2343_234375

theorem vector_sum_example : 
  (5 : ℝ) • (1 : Fin 3 → ℝ) 0 + (-3 : ℝ) • (1 : Fin 3 → ℝ) 1 + (2 : ℝ) • (1 : Fin 3 → ℝ) 2 + 
  (-4 : ℝ) • (1 : Fin 3 → ℝ) 0 + (8 : ℝ) • (1 : Fin 3 → ℝ) 1 + (-1 : ℝ) • (1 : Fin 3 → ℝ) 2 = 
  (1 : ℝ) • (1 : Fin 3 → ℝ) 0 + (5 : ℝ) • (1 : Fin 3 → ℝ) 1 + (1 : ℝ) • (1 : Fin 3 → ℝ) 2 :=
by sorry

end vector_sum_example_l2343_234375


namespace impossible_to_use_all_components_l2343_234302

theorem impossible_to_use_all_components (p q r : ℤ) : 
  ¬ ∃ (x y z : ℤ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end impossible_to_use_all_components_l2343_234302


namespace minimum_boxes_required_l2343_234358

structure BoxType where
  capacity : ℕ
  quantity : ℕ

def total_brochures : ℕ := 10000

def small_box : BoxType := ⟨50, 40⟩
def medium_box : BoxType := ⟨200, 25⟩
def large_box : BoxType := ⟨500, 10⟩

def box_types : List BoxType := [small_box, medium_box, large_box]

def can_ship (boxes : List (BoxType × ℕ)) : Prop :=
  (boxes.map (λ (b, n) => b.capacity * n)).sum ≥ total_brochures

theorem minimum_boxes_required :
  ∃ (boxes : List (BoxType × ℕ)),
    (boxes.map Prod.snd).sum = 35 ∧
    can_ship boxes ∧
    ∀ (other_boxes : List (BoxType × ℕ)),
      can_ship other_boxes →
      (other_boxes.map Prod.snd).sum ≥ 35 :=
sorry

end minimum_boxes_required_l2343_234358


namespace henrys_scores_l2343_234309

theorem henrys_scores (G M : ℝ) (h1 : G + M + 66 + (G + M + 66) / 3 = 248) : G + M = 120 := by
  sorry

end henrys_scores_l2343_234309


namespace military_unit_reorganization_l2343_234314

theorem military_unit_reorganization (x : ℕ) : 
  (x * (x + 5) = 5 * (x + 845)) → 
  (x * (x + 5) = 4550) := by
  sorry

end military_unit_reorganization_l2343_234314


namespace arithmetic_sequence_twelfth_term_l2343_234369

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequenceTerm (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_twelfth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 13)
  (h_seventh : a 7 = 25) :
  a 12 = 40 := by
  sorry

end arithmetic_sequence_twelfth_term_l2343_234369


namespace longer_show_episode_length_l2343_234339

/-- Given two TV shows, prove the length of each episode of the longer show -/
theorem longer_show_episode_length 
  (total_watch_time : ℝ)
  (short_show_episode_length : ℝ)
  (short_show_episodes : ℕ)
  (long_show_episodes : ℕ)
  (h1 : total_watch_time = 24)
  (h2 : short_show_episode_length = 0.5)
  (h3 : short_show_episodes = 24)
  (h4 : long_show_episodes = 12) :
  (total_watch_time - short_show_episode_length * short_show_episodes) / long_show_episodes = 1 := by
sorry

end longer_show_episode_length_l2343_234339


namespace hyperbola_equation_l2343_234363

-- Define the hyperbola
def Hyperbola (x y : ℝ) := y^2 - x^2/2 = 1

-- Define the asymptotes
def Asymptotes (x y : ℝ) := (x + Real.sqrt 2 * y = 0) ∨ (x - Real.sqrt 2 * y = 0)

theorem hyperbola_equation :
  ∀ (x y : ℝ),
  Asymptotes x y →
  Hyperbola (-2) (Real.sqrt 3) →
  Hyperbola x y :=
sorry

end hyperbola_equation_l2343_234363


namespace ap_has_twelve_terms_l2343_234350

/-- Represents an arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                  -- number of terms
  first_term : ℝ         -- first term
  last_term : ℝ          -- last term
  odd_sum : ℝ            -- sum of odd-numbered terms
  even_sum : ℝ           -- sum of even-numbered terms
  even_terms : Even n    -- n is even
  first_term_eq : first_term = 3
  last_term_diff : last_term = first_term + 22.5
  odd_sum_eq : odd_sum = 42
  even_sum_eq : even_sum = 48

/-- Theorem stating that the arithmetic progression satisfying given conditions has 12 terms -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end ap_has_twelve_terms_l2343_234350


namespace quadratic_root_difference_l2343_234320

/-- Given a quadratic equation 5x^2 - 11x - 14 = 0, prove that the positive difference
    between its roots is √401/5 and that p + q = 406 --/
theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 5
  let b : ℝ := -11
  let c : ℝ := -14
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  let difference := |root1 - root2|
  difference = Real.sqrt 401 / 5 ∧ 401 + 5 = 406 :=
by sorry

end quadratic_root_difference_l2343_234320


namespace student_count_l2343_234345

theorem student_count (total_eggs : ℕ) (eggs_per_student : ℕ) (h1 : total_eggs = 56) (h2 : eggs_per_student = 8) :
  total_eggs / eggs_per_student = 7 := by
  sorry

end student_count_l2343_234345


namespace transformation_result_l2343_234396

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The final point after transformations -/
def final_point : ℝ × ℝ := (2, 4)

theorem transformation_result :
  translate_right (reflect_x (reflect_y initial_point)) 5 = final_point := by
  sorry

end transformation_result_l2343_234396


namespace greg_pages_per_day_l2343_234331

/-- 
Given that Brad reads 26 pages per day and 8 more pages than Greg each day,
prove that Greg reads 18 pages per day.
-/
theorem greg_pages_per_day 
  (brad_pages : ℕ) 
  (difference : ℕ) 
  (h1 : brad_pages = 26)
  (h2 : difference = 8)
  : brad_pages - difference = 18 := by
  sorry

end greg_pages_per_day_l2343_234331


namespace mrs_blue_orchard_yield_l2343_234352

/-- Calculates the expected apple yield from a rectangular orchard -/
def expected_apple_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_ft := length_steps * step_length
  let width_ft := width_steps * step_length
  let area_sqft := length_ft * width_ft
  area_sqft * yield_per_sqft

/-- Theorem stating the expected apple yield for Mrs. Blue's orchard -/
theorem mrs_blue_orchard_yield :
  expected_apple_yield 25 20 2.5 0.75 = 2343.75 := by
  sorry

end mrs_blue_orchard_yield_l2343_234352


namespace line_passes_through_fixed_point_l2343_234305

/-- Proves that the line y - 2 = mx + m passes through the point (-1, 2) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  2 - 2 = m * (-1) + m := by sorry

end line_passes_through_fixed_point_l2343_234305


namespace sqrt_three_product_l2343_234361

theorem sqrt_three_product : 5 * Real.sqrt 3 * (2 * Real.sqrt 3) = 30 := by
  sorry

end sqrt_three_product_l2343_234361


namespace largest_inscribed_circle_l2343_234343

/-- The plane region defined by the given inequalities -/
def PlaneRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 4 * p.1 + 3 * p.2 - 12 ≤ 0}

/-- The circle with center (1,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The circle is inscribed in the plane region -/
def IsInscribed (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  c ⊆ r ∧ ∃ p q s : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ q ∈ c ∧ q ∈ r ∧ s ∈ c ∧ s ∈ r ∧
    p.1 = 0 ∧ q.2 = 0 ∧ 4 * s.1 + 3 * s.2 = 12

/-- The circle is the largest inscribed circle -/
theorem largest_inscribed_circle :
  IsInscribed Circle PlaneRegion ∧
  ∀ c : Set (ℝ × ℝ), IsInscribed c PlaneRegion → MeasureTheory.volume c ≤ MeasureTheory.volume Circle :=
by sorry

end largest_inscribed_circle_l2343_234343


namespace quadratic_radicals_same_type_l2343_234371

theorem quadratic_radicals_same_type (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 := by
  sorry

end quadratic_radicals_same_type_l2343_234371


namespace volume_per_balloon_l2343_234391

/-- Given the number of balloons, volume of each gas tank, and number of tanks needed,
    prove that the volume of air per balloon is 10 liters. -/
theorem volume_per_balloon
  (num_balloons : ℕ)
  (tank_volume : ℕ)
  (num_tanks : ℕ)
  (h1 : num_balloons = 1000)
  (h2 : tank_volume = 500)
  (h3 : num_tanks = 20) :
  (num_tanks * tank_volume) / num_balloons = 10 :=
by sorry

end volume_per_balloon_l2343_234391


namespace employees_in_all_restaurants_l2343_234306

/-- The number of employees trained to work in all three restaurants at a resort --/
theorem employees_in_all_restaurants (total_employees : ℕ) 
  (family_buffet dining_room snack_bar in_two_restaurants : ℕ) : 
  total_employees = 39 →
  family_buffet = 19 →
  dining_room = 18 →
  snack_bar = 12 →
  in_two_restaurants = 4 →
  ∃ (in_all_restaurants : ℕ),
    family_buffet + dining_room + snack_bar - in_two_restaurants - 2 * in_all_restaurants = total_employees ∧
    in_all_restaurants = 5 :=
by sorry

end employees_in_all_restaurants_l2343_234306


namespace expression_factorization_l2343_234323

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 - 100 * x^2 + 90 * x - 10) - (5 * x^3 - 10 * x^2 + 5) = 
  15 * (x^3 - 6 * x^2 + 6 * x - 1) := by
  sorry

end expression_factorization_l2343_234323


namespace range_of_m_when_not_two_distinct_positive_roots_l2343_234397

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the condition for two distinct positive real roots
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- The theorem to prove
theorem range_of_m_when_not_two_distinct_positive_roots :
  {m : ℝ | ¬(has_two_distinct_positive_roots m)} = Set.Ici (-2) :=
by sorry

end range_of_m_when_not_two_distinct_positive_roots_l2343_234397


namespace quadrilateral_area_l2343_234330

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AB : ℝ) (BC : ℝ) (AD : ℝ) (DC : ℝ)
  (AB_perp_BC : AB = BC)
  (AD_perp_DC : AD = DC)
  (AB_eq_9 : AB = 9)
  (AD_eq_8 : AD = 8)

/-- The area of the quadrilateral ABCD is 82.5 square units -/
theorem quadrilateral_area (q : Quadrilateral) : Real.sqrt ((q.AB ^ 2 + q.BC ^ 2) * (q.AD ^ 2 + q.DC ^ 2)) / 2 = 82.5 := by
  sorry

#check quadrilateral_area

end quadrilateral_area_l2343_234330


namespace conference_handshakes_l2343_234370

/-- The number of companies at the conference -/
def num_companies : ℕ := 5

/-- The number of representatives per company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the conference -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company

/-- The total number of handshakes at the conference -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem conference_handshakes : total_handshakes = 250 := by
  sorry

end conference_handshakes_l2343_234370


namespace leader_assistant_selection_l2343_234344

theorem leader_assistant_selection (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end leader_assistant_selection_l2343_234344


namespace sqrt_meaningful_range_l2343_234384

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 4) → x ≥ -4 := by
  sorry

end sqrt_meaningful_range_l2343_234384


namespace right_triangle_hypotenuse_l2343_234318

theorem right_triangle_hypotenuse (DE DF : ℝ) (P Q : ℝ × ℝ) :
  DE > 0 →
  DF > 0 →
  P.1 = DE / 4 →
  P.2 = 0 →
  Q.1 = 0 →
  Q.2 = DF / 4 →
  (DE - P.1)^2 + DF^2 = 18^2 →
  DE^2 + (DF - Q.2)^2 = 30^2 →
  DE^2 + DF^2 = (24 * Real.sqrt 3)^2 := by
sorry

end right_triangle_hypotenuse_l2343_234318


namespace alpha_monogram_count_l2343_234328

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of possible monograms for baby Alpha -/
def num_monograms : ℕ := n.choose k

theorem alpha_monogram_count : num_monograms = 300 := by
  sorry

end alpha_monogram_count_l2343_234328


namespace inverse_quadratic_equation_l2343_234351

theorem inverse_quadratic_equation (x : ℝ) :
  (1 : ℝ) = 1 / (3 * x^2 + 2 * x + 1) → x = 0 ∨ x = -2/3 := by
  sorry

end inverse_quadratic_equation_l2343_234351


namespace bacteria_growth_l2343_234374

/-- Calculates the bacteria population after a given time -/
def bacteria_population (initial_count : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_count * 2^(elapsed_time / doubling_time)

/-- Theorem: The bacteria population after 20 minutes is 240 -/
theorem bacteria_growth : bacteria_population 15 5 20 = 240 := by
  sorry

end bacteria_growth_l2343_234374


namespace inequality_equivalence_l2343_234340

theorem inequality_equivalence (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -4/3) :
  (x + 3) / (x - 1) > (4 * x + 5) / (3 * x + 4) ↔ 7 - Real.sqrt 66 < x ∧ x < 7 + Real.sqrt 66 :=
by sorry

end inequality_equivalence_l2343_234340


namespace arithmetic_sequence_sum_l2343_234390

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The sum of specific terms in the sequence equals 20 -/
def SumEquals20 (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals20 a) : 
  a 1 + a 13 = 8 := by
sorry

end arithmetic_sequence_sum_l2343_234390


namespace trapezoid_BE_length_l2343_234356

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a trapezoid ABCD with point F outside -/
structure Trapezoid :=
  (A B C D F : Point)
  (is_trapezoid : sorry)  -- Condition that ABCD is a trapezoid
  (F_on_AD_extension : sorry)  -- Condition that F is on the extension of AD

/-- Given a trapezoid, find point E on AC such that E is on BF -/
def find_E (t : Trapezoid) : Point :=
  sorry

/-- Given a trapezoid, find point G on the extension of DC such that FG is parallel to BC -/
def find_G (t : Trapezoid) : Point :=
  sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem trapezoid_BE_length (t : Trapezoid) :
  let E := find_E t
  let G := find_G t
  distance t.B E = 30 :=
  sorry

end trapezoid_BE_length_l2343_234356


namespace hockey_players_count_l2343_234365

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ) 
  (h1 : total_players = 55)
  (h2 : cricket_players = 15)
  (h3 : football_players = 13)
  (h4 : softball_players = 15) :
  total_players - (cricket_players + football_players + softball_players) = 12 :=
by
  sorry

end hockey_players_count_l2343_234365


namespace f_monotonicity_and_m_range_l2343_234327

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x - 1

theorem f_monotonicity_and_m_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧ 
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ m : ℝ, (∀ a : ℝ, -1 < a ∧ a < 1 → ∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ m * a - f x₀ < 0) ↔ 
    -1 / Real.exp 1 ≤ m ∧ m ≤ 1 / Real.exp 1) :=
sorry

end

end f_monotonicity_and_m_range_l2343_234327


namespace tom_profit_l2343_234367

/-- Calculates the profit from a stock transaction -/
def calculate_profit (
  initial_shares : ℕ
  ) (initial_price : ℚ
  ) (sold_shares : ℕ
  ) (selling_price : ℚ
  ) (remaining_shares_value_multiplier : ℚ
  ) : ℚ :=
  let total_cost := initial_shares * initial_price
  let revenue_from_sold := sold_shares * selling_price
  let revenue_from_remaining := (initial_shares - sold_shares) * (initial_price * remaining_shares_value_multiplier)
  let total_revenue := revenue_from_sold + revenue_from_remaining
  total_revenue - total_cost

/-- Tom's stock transaction profit is $40 -/
theorem tom_profit : 
  calculate_profit 20 3 10 4 2 = 40 := by
  sorry

end tom_profit_l2343_234367


namespace rectangle_area_rectangle_area_proof_l2343_234362

/-- Given two adjacent vertices of a rectangle at (-3, 2) and (1, -6),
    with the third vertex forming a right angle at (-3, 2) and
    the fourth vertex aligning vertically with (-3, 2),
    prove that the area of the rectangle is 32√5. -/
theorem rectangle_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (-3, 2)
    let v2 : ℝ × ℝ := (1, -6)
    let v3 : ℝ × ℝ := (-3, 2 - Real.sqrt 80)
    let v4 : ℝ × ℝ := (-3, -6)
    (v1.1 = v3.1 ∧ v1.1 = v4.1) →  -- fourth vertex aligns vertically with (-3, 2)
    (v1.2 - v3.2)^2 + (v1.1 - v2.1)^2 = (v1.2 - v2.2)^2 →  -- right angle at (-3, 2)
    area = 32 * Real.sqrt 5

-- The proof of this theorem
theorem rectangle_area_proof : rectangle_area (32 * Real.sqrt 5) := by
  sorry

end rectangle_area_rectangle_area_proof_l2343_234362


namespace chef_eggs_proof_l2343_234311

def initial_eggs (eggs_in_fridge : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) : ℕ :=
  eggs_in_fridge + eggs_per_cake * num_cakes

theorem chef_eggs_proof :
  initial_eggs 10 5 10 = 60 := by
  sorry

end chef_eggs_proof_l2343_234311


namespace largest_product_of_three_primes_digit_sum_l2343_234394

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    d < 10 ∧
    e < 10 ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') → 
      is_prime d' ∧ 
      is_prime e' ∧ 
      is_prime (10 * d' + e') ∧ 
      d' < 10 ∧ 
      e' < 10 → 
      m ≤ n) ∧
    sum_of_digits n = 12 :=
sorry

end largest_product_of_three_primes_digit_sum_l2343_234394


namespace lowest_cost_plan_l2343_234368

/-- Represents a plan for setting up reading corners --/
structure ReadingCornerPlan where
  medium : ℕ
  small : ℕ

/-- Checks if a plan satisfies the book constraints --/
def satisfiesBookConstraints (plan : ReadingCornerPlan) : Prop :=
  plan.medium * 80 + plan.small * 30 ≤ 1900 ∧
  plan.medium * 50 + plan.small * 60 ≤ 1620

/-- Checks if a plan satisfies the total number of corners constraint --/
def satisfiesTotalCorners (plan : ReadingCornerPlan) : Prop :=
  plan.medium + plan.small = 30

/-- Calculates the total cost of a plan --/
def totalCost (plan : ReadingCornerPlan) : ℕ :=
  plan.medium * 860 + plan.small * 570

/-- The theorem to be proved --/
theorem lowest_cost_plan :
  ∃ (plan : ReadingCornerPlan),
    satisfiesBookConstraints plan ∧
    satisfiesTotalCorners plan ∧
    plan.medium = 18 ∧
    plan.small = 12 ∧
    totalCost plan = 22320 ∧
    ∀ (other : ReadingCornerPlan),
      satisfiesBookConstraints other →
      satisfiesTotalCorners other →
      totalCost plan ≤ totalCost other :=
  sorry

end lowest_cost_plan_l2343_234368


namespace intersection_M_N_l2343_234359

def M : Set ℝ := {x | (x + 1) * (x - 3) < 0}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_M_N_l2343_234359


namespace min_type_A_buses_l2343_234308

/-- Represents the number of Type A buses -/
def x : ℕ := sorry

/-- The capacity of a Type A bus -/
def capacity_A : ℕ := 45

/-- The capacity of a Type B bus -/
def capacity_B : ℕ := 30

/-- The total number of people to transport -/
def total_people : ℕ := 300

/-- The total number of buses to be rented -/
def total_buses : ℕ := 8

/-- The minimum number of Type A buses needed -/
def min_buses_A : ℕ := 4

theorem min_type_A_buses :
  (∀ n : ℕ, n ≥ min_buses_A →
    capacity_A * n + capacity_B * (total_buses - n) ≥ total_people) ∧
  (∀ m : ℕ, m < min_buses_A →
    capacity_A * m + capacity_B * (total_buses - m) < total_people) :=
by sorry

end min_type_A_buses_l2343_234308


namespace unit_digit_of_fraction_l2343_234392

theorem unit_digit_of_fraction (n : ℕ) :
  (33 * 10) / (2^1984) % 10 = 6 := by sorry

end unit_digit_of_fraction_l2343_234392


namespace force_on_smooth_surface_with_pulleys_l2343_234342

/-- The force required to move a mass on a smooth horizontal surface using a pulley system -/
theorem force_on_smooth_surface_with_pulleys 
  (m : ℝ) -- mass in kg
  (g : ℝ) -- acceleration due to gravity in m/s²
  (h_m_pos : m > 0)
  (h_g_pos : g > 0) :
  ∃ F : ℝ, F = 4 * m * g :=
by sorry

end force_on_smooth_surface_with_pulleys_l2343_234342


namespace expression_equality_l2343_234322

theorem expression_equality : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end expression_equality_l2343_234322


namespace product_trailing_zeros_l2343_234379

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem product_trailing_zeros :
  trailing_zeros 100 = 24 :=
sorry

end product_trailing_zeros_l2343_234379


namespace tax_percentage_calculation_l2343_234357

def annual_salary : ℝ := 40000
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800
def take_home_pay : ℝ := 27200

theorem tax_percentage_calculation :
  let healthcare_deduction := annual_salary * healthcare_rate
  let total_non_tax_deductions := healthcare_deduction + union_dues
  let amount_before_taxes := annual_salary - total_non_tax_deductions
  let tax_deduction := amount_before_taxes - take_home_pay
  let tax_percentage := (tax_deduction / annual_salary) * 100
  tax_percentage = 20 := by sorry

end tax_percentage_calculation_l2343_234357


namespace max_boxes_in_lot_l2343_234319

theorem max_boxes_in_lot (lot_width lot_length box_width box_length : ℕ) 
  (hw : lot_width = 36)
  (hl : lot_length = 72)
  (bw : box_width = 3)
  (bl : box_length = 4) :
  (lot_width / box_width) * (lot_length / box_length) = 216 := by
  sorry

end max_boxes_in_lot_l2343_234319


namespace telescope_visual_range_l2343_234386

/-- 
Given a telescope that increases the visual range by 50% to reach 150 kilometers,
prove that the initial visual range without the telescope was 100 kilometers.
-/
theorem telescope_visual_range 
  (increased_range : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increased_range = 150) 
  (h2 : increase_percentage = 0.5) : 
  increased_range / (1 + increase_percentage) = 100 := by
  sorry

end telescope_visual_range_l2343_234386


namespace max_checkers_on_6x6_board_l2343_234326

/-- A checker placement on a 6x6 board is represented as a list of 36 booleans -/
def CheckerPlacement := List Bool

/-- A function to check if three points are collinear on a 6x6 board -/
def areCollinear (p1 p2 p3 : Nat × Nat) : Bool :=
  sorry

/-- A function to check if a placement is valid (no three checkers are collinear) -/
def isValidPlacement (placement : CheckerPlacement) : Bool :=
  sorry

/-- The maximum number of checkers that can be placed on a 6x6 board
    such that no three checkers are collinear -/
def maxCheckers : Nat := 12

/-- Theorem stating that 12 is the maximum number of checkers
    that can be placed on a 6x6 board with no three collinear -/
theorem max_checkers_on_6x6_board :
  (∀ placement : CheckerPlacement,
    isValidPlacement placement → placement.length ≤ maxCheckers) ∧
  (∃ placement : CheckerPlacement,
    isValidPlacement placement ∧ placement.length = maxCheckers) :=
sorry

end max_checkers_on_6x6_board_l2343_234326


namespace square_area_ratio_l2343_234372

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = a * Real.sqrt 3) :
  b ^ 2 = 3 * a ^ 2 := by sorry

end square_area_ratio_l2343_234372


namespace triangle_side_length_l2343_234398

/-- Given an acute triangle ABC with sides a, b, and c, 
    if a = 4, b = 3, and the area is 3√3, then c = √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → 
  b = 3 → 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 →
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  c = Real.sqrt 13 := by
  sorry


end triangle_side_length_l2343_234398


namespace simplify_and_evaluate_specific_case_l2343_234337

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = a^2 - 8*b^2 :=
by sorry

theorem specific_case : 
  let a : ℝ := -1
  let b : ℝ := 2
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = -31 :=
by sorry

end simplify_and_evaluate_specific_case_l2343_234337


namespace minimum_fifth_quarter_score_l2343_234334

def required_average : ℚ := 85
def num_quarters : ℕ := 5
def first_four_scores : List ℚ := [84, 80, 78, 82]

theorem minimum_fifth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_four := first_four_scores.sum
  let min_fifth_score := total_required - sum_first_four
  min_fifth_score = 101 := by sorry

end minimum_fifth_quarter_score_l2343_234334


namespace sample_size_is_200_l2343_234373

/-- Represents a statistical survey of students -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- Definition of sample size for a student survey -/
def sample_size (survey : StudentSurvey) : ℕ := survey.selected_students

/-- Theorem stating that for the given survey, the sample size is 200 -/
theorem sample_size_is_200 (survey : StudentSurvey) 
  (h1 : survey.total_students = 2000) 
  (h2 : survey.selected_students = 200) : 
  sample_size survey = 200 := by
  sorry

#check sample_size_is_200

end sample_size_is_200_l2343_234373


namespace absolute_value_equation_l2343_234355

theorem absolute_value_equation (a b c : ℝ) :
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) →
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by sorry

end absolute_value_equation_l2343_234355


namespace geometric_sequence_sum_property_l2343_234349

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := k + 3^n

/-- Definition of a term in the sequence -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

/-- Theorem stating that k = -1 for the given conditions -/
theorem geometric_sequence_sum_property (k : ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n+1) k / a n k = a (n+2) k / a (n+1) k) →
  k = -1 :=
sorry

end geometric_sequence_sum_property_l2343_234349


namespace scale_length_l2343_234335

/-- The total length of a scale divided into equal parts -/
def total_length (num_parts : ℕ) (part_length : ℝ) : ℝ :=
  num_parts * part_length

/-- Theorem: The total length of a scale is 80 inches -/
theorem scale_length : total_length 4 20 = 80 := by
  sorry

end scale_length_l2343_234335


namespace remainder_2365947_div_8_l2343_234348

theorem remainder_2365947_div_8 : 2365947 % 8 = 3 := by sorry

end remainder_2365947_div_8_l2343_234348


namespace triangle_inequality_sum_l2343_234315

/-- Given a triangle with side lengths a, b, c, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_sum_l2343_234315


namespace angle_PQT_measure_l2343_234389

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle PQT in a regular octagon -/
def angle_PQT (octagon : RegularOctagon) : ℝ :=
  22.5

theorem angle_PQT_measure (octagon : RegularOctagon) :
  angle_PQT octagon = 22.5 := by
  sorry

end angle_PQT_measure_l2343_234389


namespace sufficient_not_necessary_condition_l2343_234383

theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (∀ a, a ≥ 1 → a + 1/a ≥ 2) ∧
  (∃ a, 0 < a ∧ a < 1 ∧ a + 1/a ≥ 2) :=
by sorry

end sufficient_not_necessary_condition_l2343_234383


namespace margo_pairing_probability_l2343_234346

/-- The probability of a specific pairing in a class with random pairings -/
def pairingProbability (totalStudents : ℕ) (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / (totalStudents - 1)

/-- Theorem: The probability of Margo being paired with either Irma or Julia -/
theorem margo_pairing_probability :
  let totalStudents : ℕ := 32
  let favorableOutcomes : ℕ := 2
  pairingProbability totalStudents favorableOutcomes = 2 / 31 := by
  sorry

end margo_pairing_probability_l2343_234346


namespace fraction_of_as_l2343_234376

theorem fraction_of_as (total : ℝ) (as : ℝ) (bs : ℝ) : 
  total > 0 → 
  bs / total = 0.2 → 
  (as + bs) / total = 0.9 → 
  as / total = 0.7 := by
sorry

end fraction_of_as_l2343_234376


namespace inequality_solution_set_l2343_234329

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) * (x + 2) < 0

-- Define the solution set
def solution_set : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Theorem statement
theorem inequality_solution_set : 
  { x : ℝ | inequality x } = solution_set := by sorry

end inequality_solution_set_l2343_234329


namespace range_of_m_l2343_234303

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - 2*x + 2 ≠ m) → m < 1 := by
  sorry

end range_of_m_l2343_234303


namespace elmer_pond_maturation_rate_l2343_234341

/-- The rate at which pollywogs mature and leave the pond -/
def maturation_rate (
  initial_pollywogs : ℕ
  ) (
  days_to_disappear : ℕ
  ) (
  catch_rate : ℕ
  ) (
  catch_days : ℕ
  ) : ℚ :=
  (initial_pollywogs - catch_rate * catch_days) / days_to_disappear

/-- Theorem stating the maturation rate of pollywogs in Elmer's pond -/
theorem elmer_pond_maturation_rate :
  maturation_rate 2400 44 10 20 = 50 := by
  sorry

end elmer_pond_maturation_rate_l2343_234341


namespace geometric_series_sum_specific_geometric_series_sum_l2343_234310

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a ≠ 0 → 
  |r| < 1 → 
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum : 
  (∑' n, (1 : ℝ) * (1/4 : ℝ)^n) = 4/3 :=
sorry

end geometric_series_sum_specific_geometric_series_sum_l2343_234310


namespace geometric_sequence_property_l2343_234347

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3 →
  a 1 * a 11 = 100 := by
  sorry

end geometric_sequence_property_l2343_234347


namespace largest_digit_divisible_by_six_l2343_234324

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3672 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (3672 * 10 + M) % 6 = 0 → M ≤ N :=
by
  -- The proof goes here
  sorry

end largest_digit_divisible_by_six_l2343_234324


namespace necessary_unique_letters_count_l2343_234378

def word : String := "necessary"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem necessary_unique_letters_count :
  (unique_letters word).card = 7 := by sorry

end necessary_unique_letters_count_l2343_234378


namespace range_of_m_l2343_234364

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (25 - m) + y^2 / (m - 7) = 1 ∧ 
  (25 - m > 0) ∧ (m - 7 > 0) ∧ (25 - m > m - 7)

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ (e : ℝ), (∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1) ∧ 
  1 < e ∧ e < 2 ∧ e^2 = (5 + m) / 5

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(¬(p m) ∨ ¬(q m))) → (7 < m ∧ m < 15) :=
sorry

end range_of_m_l2343_234364


namespace compound_propositions_truth_l2343_234332

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Theorem statement
theorem compound_propositions_truth (hp : p) (hq : ¬q) : 
  (p ∧ q = False) ∧ 
  (p ∨ q = True) ∧ 
  (p ∧ (¬q) = True) ∧ 
  ((¬p) ∨ q = False) := by
  sorry

end compound_propositions_truth_l2343_234332


namespace jelly_bean_probability_l2343_234382

/-- The number of red jelly beans -/
def red_beans : ℕ := 10

/-- The number of green jelly beans -/
def green_beans : ℕ := 12

/-- The number of yellow jelly beans -/
def yellow_beans : ℕ := 13

/-- The number of blue jelly beans -/
def blue_beans : ℕ := 15

/-- The number of purple jelly beans -/
def purple_beans : ℕ := 5

/-- The total number of jelly beans -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The number of blue and purple jelly beans combined -/
def blue_and_purple : ℕ := blue_beans + purple_beans

/-- The probability of selecting either a blue or purple jelly bean -/
def probability : ℚ := blue_and_purple / total_beans

theorem jelly_bean_probability : probability = 4 / 11 := by
  sorry

end jelly_bean_probability_l2343_234382


namespace arithmetic_sequence_count_l2343_234304

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 165 ∧ aₙ = 40 ∧ d = -5 ∧ aₙ = a₁ + (n - 1) * d → n = 26 := by
  sorry

end arithmetic_sequence_count_l2343_234304


namespace difference_is_nine_l2343_234395

theorem difference_is_nine (a b c d : ℝ) 
  (h1 : ∃ x, a - b = c + d + x)
  (h2 : a + b = c - d - 3)
  (h3 : a - c = 3) :
  (a - b) - (c + d) = 9 := by
  sorry

end difference_is_nine_l2343_234395


namespace candy_bar_cost_after_tax_l2343_234354

-- Define the initial amount Peter has
def initial_amount : ℝ := 10

-- Define the cost per ounce of soda
def soda_cost_per_ounce : ℝ := 0.25

-- Define the number of ounces of soda bought
def soda_ounces : ℝ := 16

-- Define the original price of chips
def chips_original_price : ℝ := 2.50

-- Define the discount rate for chips
def chips_discount_rate : ℝ := 0.1

-- Define the price of the candy bar
def candy_bar_price : ℝ := 1.25

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the discounted price of chips
def discounted_chips_price : ℝ := chips_original_price * (1 - chips_discount_rate)

-- Define the function to calculate the total cost before tax
def total_cost_before_tax : ℝ := soda_cost_per_ounce * soda_ounces + discounted_chips_price + candy_bar_price

-- Define the function to calculate the total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax * (1 + sales_tax_rate)

-- Theorem: The cost of the candy bar after tax is $1.35
theorem candy_bar_cost_after_tax :
  candy_bar_price * (1 + sales_tax_rate) = 1.35 ∧ total_cost_after_tax = initial_amount :=
by sorry

end candy_bar_cost_after_tax_l2343_234354


namespace stating_assignment_ways_l2343_234313

/-- Represents the number of student volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of posts -/
def num_posts : ℕ := 4

/-- Represents the number of ways A and B can be assigned to posts -/
def ways_to_assign_A_and_B : ℕ := num_posts * (num_posts - 1)

/-- 
Theorem stating that the number of ways for A and B to each independently 
take charge of one post, while ensuring each post is staffed by at least 
one volunteer, is equal to 72.
-/
theorem assignment_ways : 
  ∃ (f : ℕ → ℕ → ℕ), 
    f ways_to_assign_A_and_B (num_volunteers - 2) = 72 ∧ 
    (∀ x y, f x y ≤ x * (y^(num_posts - 2))) := by
  sorry

end stating_assignment_ways_l2343_234313


namespace fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2343_234338

/-- Definition of the function f(x) -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- Definition of a fixed point -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point 2 (-2) x₁ ∧ is_fixed_point 2 (-2) x₂ ∧ x₁ = -1 ∧ x₂ = 2 := by
  sorry

theorem range_of_a_for_two_fixed_points :
  ∀ (a : ℝ), (∀ (b : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 2) := by
  sorry

end fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2343_234338


namespace sum_of_three_numbers_l2343_234388

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  c = 2 * a →
  a + b + c = 115 := by
sorry

end sum_of_three_numbers_l2343_234388


namespace power_of_i_sum_l2343_234387

theorem power_of_i_sum (i : ℂ) : i^2 = -1 → i^44 + i^444 + 3 = 5 := by
  sorry

end power_of_i_sum_l2343_234387


namespace track_duration_in_seconds_l2343_234399

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℚ) : ℚ := minutes * 60

/-- The duration of the music track in minutes -/
def trackDurationMinutes : ℚ := 12.5

/-- Theorem: A music track playing for 12.5 minutes lasts 750 seconds -/
theorem track_duration_in_seconds : 
  minutesToSeconds trackDurationMinutes = 750 := by sorry

end track_duration_in_seconds_l2343_234399


namespace train_length_l2343_234333

/-- The length of a train given specific crossing times -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 120)
  (h2 : platform_crossing_time = 240)
  (h3 : platform_length = 1200) : 
  ∃ (train_length : ℝ), train_length = 1200 ∧ 
    (train_length / tree_crossing_time) * platform_crossing_time = train_length + platform_length :=
by
  sorry

end train_length_l2343_234333


namespace largest_power_dividing_product_l2343_234321

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define the product of pow(n) from 2 to 7000
def product : ℕ :=
  sorry

-- State the theorem
theorem largest_power_dividing_product :
  ∃ m : ℕ, (4620 ^ m : ℕ) ∣ product ∧
  ∀ k : ℕ, (4620 ^ k : ℕ) ∣ product → k ≤ m ∧
  m = 698 :=
sorry

end largest_power_dividing_product_l2343_234321


namespace used_car_clients_l2343_234317

theorem used_car_clients (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  total_cars = 12 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  (total_cars * selections_per_car) / cars_per_client = 9 :=
by sorry

end used_car_clients_l2343_234317


namespace point_distance_ratio_l2343_234381

theorem point_distance_ratio (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  (P.1)^2 + (P.2)^2 = 10^2 → 
  (abs P.2) / 10 = 1 / 2 := by sorry

end point_distance_ratio_l2343_234381


namespace smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l2343_234312

theorem smallest_integer_solution_inequality :
  ∀ x : ℤ, (9*x + 8)/6 - x/3 ≥ -1 → x ≥ -2 :=
by
  sorry

theorem neg_two_satisfies_inequality :
  (9*(-2) + 8)/6 - (-2)/3 ≥ -1 :=
by
  sorry

theorem neg_two_is_smallest_integer_solution :
  ∀ y : ℤ, y < -2 → (9*y + 8)/6 - y/3 < -1 :=
by
  sorry

end smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l2343_234312


namespace definite_integrals_l2343_234325

theorem definite_integrals :
  (∫ (x : ℝ) in (-1)..(1), x^3) = 0 ∧
  (∫ (x : ℝ) in (2)..(ℯ + 1), 1 / (x - 1)) = 1 := by
  sorry

end definite_integrals_l2343_234325


namespace tan_sum_pi_12_pi_4_l2343_234380

theorem tan_sum_pi_12_pi_4 : 
  Real.tan (π / 12) + Real.tan (π / 4) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by sorry

end tan_sum_pi_12_pi_4_l2343_234380


namespace rhombus_diagonal_l2343_234300

/-- Given a rhombus with one diagonal of 65 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 60 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (area : ℝ) (d₂ : ℝ) : 
  d₁ = 65 → area = 1950 → area = (d₁ * d₂) / 2 → d₂ = 60 := by
sorry

end rhombus_diagonal_l2343_234300


namespace final_position_theorem_l2343_234353

/-- Represents the position of the letter L --/
inductive LPosition
  | PosXPosY  -- Base along positive x-axis, stem along positive y-axis
  | NegXPosY  -- Base along negative x-axis, stem along positive y-axis
  | PosXNegY  -- Base along positive x-axis, stem along negative y-axis
  | NegXNegY  -- Base along negative x-axis, stem along negative y-axis

/-- Represents the transformations --/
inductive Transformation
  | RotateClockwise180
  | ReflectXAxis
  | RotateHalfTurn
  | ReflectYAxis

/-- Applies a single transformation to a given position --/
def applyTransformation (pos : LPosition) (t : Transformation) : LPosition :=
  sorry

/-- Applies a sequence of transformations to a given position --/
def applyTransformations (pos : LPosition) (ts : List Transformation) : LPosition :=
  sorry

theorem final_position_theorem :
  let initialPos := LPosition.PosXPosY
  let transformations := [
    Transformation.RotateClockwise180,
    Transformation.ReflectXAxis,
    Transformation.RotateHalfTurn,
    Transformation.ReflectYAxis
  ]
  applyTransformations initialPos transformations = LPosition.NegXNegY :=
sorry

end final_position_theorem_l2343_234353


namespace factorial_sum_quotient_l2343_234336

theorem factorial_sum_quotient (n : ℕ) (h : n ≥ 2) :
  (Nat.factorial (n + 2) + Nat.factorial (n + 1)) / Nat.factorial (n + 1) = n + 3 := by
  sorry

end factorial_sum_quotient_l2343_234336


namespace garden_plants_correct_l2343_234307

/-- Calculates the total number of plants in Papi Calot's garden -/
def garden_plants : ℕ × ℕ × ℕ :=
  let potato_rows := 8
  let potato_alt1 := 22
  let potato_alt2 := 25
  let potato_extra := 18

  let carrot_rows := 12
  let carrot_start := 30
  let carrot_increment := 5
  let carrot_extra := 24

  let onion_repetitions := 4
  let onion_row1 := 15
  let onion_row2 := 20
  let onion_row3 := 25
  let onion_extra := 12

  let potatoes := (potato_rows / 2 * potato_alt1 + potato_rows / 2 * potato_alt2) + potato_extra
  let carrots := (carrot_rows * (2 * carrot_start + (carrot_rows - 1) * carrot_increment)) / 2 + carrot_extra
  let onions := onion_repetitions * (onion_row1 + onion_row2 + onion_row3) + onion_extra

  (potatoes, carrots, onions)

theorem garden_plants_correct :
  garden_plants = (206, 714, 252) :=
by sorry

end garden_plants_correct_l2343_234307


namespace jeremy_overall_accuracy_l2343_234301

theorem jeremy_overall_accuracy 
  (individual_portion : Real) 
  (collaborative_portion : Real)
  (terry_individual_accuracy : Real)
  (terry_overall_accuracy : Real)
  (jeremy_individual_accuracy : Real)
  (h1 : individual_portion = 0.6)
  (h2 : collaborative_portion = 0.4)
  (h3 : individual_portion + collaborative_portion = 1)
  (h4 : terry_individual_accuracy = 0.75)
  (h5 : terry_overall_accuracy = 0.85)
  (h6 : jeremy_individual_accuracy = 0.8) :
  jeremy_individual_accuracy * individual_portion + 
  (terry_overall_accuracy - terry_individual_accuracy * individual_portion) = 0.88 :=
sorry

end jeremy_overall_accuracy_l2343_234301


namespace total_yield_before_change_l2343_234316

theorem total_yield_before_change (x y z : ℝ) 
  (h1 : 0.4 * x + 0.2 * y = 5)
  (h2 : 0.4 * y + 0.2 * z = 10)
  (h3 : 0.4 * z + 0.2 * x = 9) :
  x + y + z = 40 := by
  sorry

end total_yield_before_change_l2343_234316
