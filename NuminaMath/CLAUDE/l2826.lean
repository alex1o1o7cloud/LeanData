import Mathlib

namespace remainder_eleven_power_2023_mod_13_l2826_282661

theorem remainder_eleven_power_2023_mod_13 : 11^2023 % 13 = 11 := by
  sorry

end remainder_eleven_power_2023_mod_13_l2826_282661


namespace average_weight_abc_l2826_282616

/-- Given the weights of three individuals a, b, and c, prove that their average weight is 45 kg -/
theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- average weight of a and b is 40 kg
  (b + c) / 2 = 47 →   -- average weight of b and c is 47 kg
  b = 39 →             -- weight of b is 39 kg
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_abc_l2826_282616


namespace median_in_third_interval_l2826_282677

/-- Represents the distribution of students across score intervals --/
structure ScoreDistribution where
  total_students : ℕ
  intervals : List ℕ
  h_total : total_students = intervals.sum

/-- The index of the interval containing the median --/
def median_interval_index (sd : ScoreDistribution) : ℕ :=
  sd.intervals.foldl
    (λ acc count =>
      if acc.1 < sd.total_students / 2 then (acc.1 + count, acc.2 + 1)
      else acc)
    (0, 0)
  |>.2

theorem median_in_third_interval (sd : ScoreDistribution) :
  sd.total_students = 100 ∧
  sd.intervals = [20, 18, 15, 22, 14, 11] →
  median_interval_index sd = 3 := by
  sorry

#eval median_interval_index ⟨100, [20, 18, 15, 22, 14, 11], rfl⟩

end median_in_third_interval_l2826_282677


namespace barium_chloride_molecular_weight_l2826_282632

/-- The molecular weight of one mole of Barium chloride, given the molecular weight of 4 moles. -/
theorem barium_chloride_molecular_weight :
  let moles : ℝ := 4
  let total_weight : ℝ := 828
  let one_mole_weight : ℝ := total_weight / moles
  one_mole_weight = 207 := by sorry

end barium_chloride_molecular_weight_l2826_282632


namespace max_value_expression_max_value_achievable_l2826_282610

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x ≤ Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

theorem max_value_achievable : 
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x = Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

end max_value_expression_max_value_achievable_l2826_282610


namespace football_team_probability_l2826_282642

/-- Given a group of 10 people with 2 from football teams and 8 from basketball teams,
    proves the probability that both randomly selected people are from football teams,
    given that one is from a football team, is 1/9. -/
theorem football_team_probability :
  let total_people : ℕ := 10
  let football_people : ℕ := 2
  let basketball_people : ℕ := 8
  let total_selections : ℕ := 9  -- Total ways to select given one is from football
  let both_football : ℕ := 1     -- Ways to select both from football given one is from football
  football_people + basketball_people = total_people →
  (both_football : ℚ) / total_selections = 1 / 9 :=
by sorry

end football_team_probability_l2826_282642


namespace smallest_number_l2826_282615

theorem smallest_number (A B C : ℚ) (hA : A = 1/2) (hB : B = 9/10) (hC : C = 2/5) :
  C ≤ A ∧ C ≤ B := by
  sorry

end smallest_number_l2826_282615


namespace sum_ways_2002_l2826_282675

/-- The number of ways to express 2002 as the sum of 3 positive integers, without considering order -/
def ways_to_sum_2002 : ℕ := 334000

/-- A function that counts the number of ways to express a given natural number as the sum of 3 positive integers, without considering order -/
def count_sum_ways (n : ℕ) : ℕ :=
  sorry

theorem sum_ways_2002 : count_sum_ways 2002 = ways_to_sum_2002 := by
  sorry

end sum_ways_2002_l2826_282675


namespace magical_red_knights_fraction_l2826_282601

theorem magical_red_knights_fraction :
  ∀ (total_knights : ℕ) (red_knights blue_knights magical_knights : ℕ) 
    (red_magical blue_magical : ℚ),
    red_knights = total_knights / 3 →
    blue_knights = total_knights - red_knights →
    magical_knights = total_knights / 5 →
    blue_magical = (2/3) * red_magical →
    red_knights * red_magical + blue_knights * blue_magical = magical_knights →
    red_magical = 9/35 := by
  sorry

end magical_red_knights_fraction_l2826_282601


namespace cubic_equation_roots_l2826_282687

/-- The cubic polynomial f(x) = x^3 + 9x^2 + 26x + 24 -/
def f (x : ℝ) : ℝ := x^3 + 9*x^2 + 26*x + 24

/-- The set of roots of f -/
def roots : Set ℝ := {x | f x = 0}

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ < 0 ∧ r₃ < 0 ∧
  roots = {r₁, r₂, r₃} :=
sorry

end cubic_equation_roots_l2826_282687


namespace root_conditions_imply_sum_l2826_282641

/-- Given two polynomial equations with specific root conditions, prove that 100p + q = 502 -/
theorem root_conditions_imply_sum (p q : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    (∀ z : ℝ, (z + p) * (z + q) * (z + 5) / (z + 2)^2 = 0 ↔ (z = x ∨ z = y)) ∧
    (z = -2 → (z + p) * (z + q) * (z + 5) ≠ 0)) →
  (∃ (u v : ℝ), u ≠ v ∧ 
    (∀ w : ℝ, (w + 2*p) * (w + 2) * (w + 3) / ((w + q) * (w + 5)) = 0 ↔ (w = u ∨ w = v)) ∧
    ((w = -q ∨ w = -5) → (w + 2*p) * (w + 2) * (w + 3) ≠ 0)) →
  100 * p + q = 502 := by
sorry

end root_conditions_imply_sum_l2826_282641


namespace min_omega_value_l2826_282672

theorem min_omega_value (ω : Real) (x₁ x₂ : Real) :
  ω > 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₁ = 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₂ = Real.sqrt 3 →
  |x₁ - x₂| = π →
  ∃ (ω_min : Real), ω_min = 1/2 ∧ ∀ (ω' : Real), ω' > 0 ∧
    (∃ (y₁ y₂ : Real), 
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₁ = 0 ∧
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₂ = Real.sqrt 3 ∧
      |y₁ - y₂| = π) →
    ω' ≥ ω_min :=
by sorry

end min_omega_value_l2826_282672


namespace tetrahedron_properties_l2826_282647

-- Define the vertices of the tetrahedron
def A1 : ℝ × ℝ × ℝ := (1, 2, 0)
def A2 : ℝ × ℝ × ℝ := (3, 0, -3)
def A3 : ℝ × ℝ × ℝ := (5, 2, 6)
def A4 : ℝ × ℝ × ℝ := (8, 4, -9)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height of a tetrahedron
def tetrahedron_height (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the specific tetrahedron
theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 34 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 + 2/7 :=
by sorry

end tetrahedron_properties_l2826_282647


namespace f_derivative_and_extrema_l2826_282653

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

theorem f_derivative_and_extrema (a : ℝ) :
  (∀ x, deriv (f a) x = 3 * x^2 - 2 * a * x - 4) ∧
  (deriv (f a) (-1) = 0 → a = 1/2) ∧
  (a = 1/2 → ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, f a x ≥ min) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = min) ∧
    max = 9/2 ∧ min = -50/27) := by
  sorry

end f_derivative_and_extrema_l2826_282653


namespace existence_of_constant_g_l2826_282628

-- Define the necessary types and functions
def Graph : Type := sorry
def circumference (G : Graph) : ℕ := sorry
def chromaticNumber (G : Graph) : ℕ := sorry
def containsSubgraph (G H : Graph) : Prop := sorry
def TK (r : ℕ) : Graph := sorry

-- The main theorem
theorem existence_of_constant_g : 
  ∃ g : ℕ, ∀ (G : Graph) (r : ℕ), 
    circumference G ≥ g → chromaticNumber G ≥ r → containsSubgraph G (TK r) := by
  sorry

end existence_of_constant_g_l2826_282628


namespace binomial_equation_solutions_l2826_282681

theorem binomial_equation_solutions :
  ∀ m r : ℕ, 2014 ≥ m → m ≥ r → r ≥ 1 →
  (Nat.choose 2014 m + Nat.choose m r = Nat.choose 2014 r + Nat.choose (2014 - r) (m - r)) ↔
  ((m = r ∧ m ≤ 2014) ∨
   (m = 2014 - r ∧ r ≤ 1006) ∨
   (m = 2014 ∧ r ≤ 2013)) :=
by sorry

end binomial_equation_solutions_l2826_282681


namespace circle_radius_with_inscribed_square_l2826_282676

/-- Given a circle with a chord of length 6 and an inscribed square of side length 2 in the segment
    corresponding to the chord, prove that the radius of the circle is √10. -/
theorem circle_radius_with_inscribed_square (r : ℝ) 
  (h1 : ∃ (chord : ℝ), chord = 6 ∧ chord ≤ 2 * r)
  (h2 : ∃ (square_side : ℝ), square_side = 2 ∧ 
        square_side ≤ (r + r - chord) ∧ 
        square_side * square_side ≤ chord * (2 * r - chord)) :
  r = Real.sqrt 10 := by
  sorry

end circle_radius_with_inscribed_square_l2826_282676


namespace short_show_episodes_count_l2826_282697

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The duration of one episode of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of one episode of the long show in hours -/
def long_show_duration : ℚ := 1

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total time Tim watched TV in hours -/
def total_watch_time : ℕ := 24

theorem short_show_episodes_count :
  short_show_episodes * short_show_duration + long_show_episodes * long_show_duration = total_watch_time := by
  sorry

end short_show_episodes_count_l2826_282697


namespace max_product_xy_l2826_282626

theorem max_product_xy (x y : ℝ) :
  (Real.sqrt (x + y - 1) + x^4 + y^4 - 1/8 ≤ 0) →
  (x * y ≤ 1/4) :=
by sorry

end max_product_xy_l2826_282626


namespace ratio_approximation_l2826_282635

/-- The set of numbers from 1 to 10^13 in powers of 10 -/
def powerSet : Set ℕ := {n | ∃ k : ℕ, k ≤ 13 ∧ n = 10^k}

/-- The largest element in the set -/
def largestElement : ℕ := 10^13

/-- The sum of all elements in the set except the largest -/
def sumOfOthers : ℕ := (largestElement - 1) / 9

/-- The ratio of the largest element to the sum of others -/
def ratio : ℚ := largestElement / sumOfOthers

theorem ratio_approximation : ∃ ε > 0, abs (ratio - 9) < ε :=
sorry

end ratio_approximation_l2826_282635


namespace range_of_m_l2826_282679

/-- A quadratic function f(x) = ax^2 - 2ax + c -/
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

/-- The statement that f is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (a c : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a c x > f a c y

/-- The main theorem -/
theorem range_of_m (a c : ℝ) :
  is_monotone_decreasing a c →
  (∃ m, f a c m ≤ f a c 0) →
  ∃ m, 0 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l2826_282679


namespace equal_division_of_cards_l2826_282638

theorem equal_division_of_cards (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) : 
  total_cards = 455 → num_friends = 5 → cards_per_friend = total_cards / num_friends → cards_per_friend = 91 := by
  sorry

end equal_division_of_cards_l2826_282638


namespace table_tennis_team_members_l2826_282602

theorem table_tennis_team_members : ∃ (x : ℕ), x > 0 ∧ x ≤ 33 ∧ 
  (∃ (s r : ℕ), s + r = x ∧ 4 * s + 3 * r + 2 * x = 33) :=
by
  sorry

end table_tennis_team_members_l2826_282602


namespace axis_of_symmetry_cos_minus_sin_l2826_282619

/-- The axis of symmetry for the function y = cos(2x) - sin(2x) is x = -π/8 -/
theorem axis_of_symmetry_cos_minus_sin (x : ℝ) : 
  (∀ y, y = Real.cos (2 * x) - Real.sin (2 * x)) → 
  (∃ k : ℤ, x = (k : ℝ) * π / 2 - π / 8) :=
by sorry

end axis_of_symmetry_cos_minus_sin_l2826_282619


namespace bobs_weekly_profit_l2826_282607

/-- Calculates the weekly profit for Bob's muffin business -/
theorem bobs_weekly_profit (muffins_per_day : ℕ) (buy_price : ℚ) (sell_price : ℚ) (days_per_week : ℕ) :
  muffins_per_day = 12 →
  buy_price = 3/4 →
  sell_price = 3/2 →
  days_per_week = 7 →
  (sell_price - buy_price) * muffins_per_day * days_per_week = 63 := by
sorry

#eval (3/2 : ℚ) - (3/4 : ℚ)
#eval ((3/2 : ℚ) - (3/4 : ℚ)) * 12
#eval (((3/2 : ℚ) - (3/4 : ℚ)) * 12) * 7

end bobs_weekly_profit_l2826_282607


namespace flowers_remaining_after_picking_l2826_282613

/-- The number of flowers remaining after Neznaika's picking --/
def remaining_flowers (total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers : ℕ) : ℕ :=
  total_flowers - unwatered_flowers - picked_tulips

/-- Theorem stating the number of remaining flowers --/
theorem flowers_remaining_after_picking 
  (total_flowers : ℕ) 
  (total_tulips : ℕ)
  (total_peonies : ℕ)
  (watered_tulips : ℕ)
  (picked_tulips : ℕ)
  (unwatered_flowers : ℕ)
  (h1 : total_flowers = 30)
  (h2 : total_tulips = 15)
  (h3 : total_peonies = 15)
  (h4 : total_flowers = total_tulips + total_peonies)
  (h5 : watered_tulips = 10)
  (h6 : unwatered_flowers = 10)
  (h7 : picked_tulips = 6)
  : remaining_flowers total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers = 19 :=
by
  sorry


end flowers_remaining_after_picking_l2826_282613


namespace range_of_a_l2826_282689

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.Iic (-2) ∪ {1} := by sorry

end range_of_a_l2826_282689


namespace C1_intersects_C2_l2826_282648

-- Define the line C1
def C1 (x : ℝ) : ℝ := 2 * x - 3

-- Define the circle C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 25

-- Theorem stating that C1 and C2 intersect
theorem C1_intersects_C2 : ∃ (x y : ℝ), y = C1 x ∧ C2 x y := by
  sorry

end C1_intersects_C2_l2826_282648


namespace brick_width_is_10cm_l2826_282637

/-- Proves that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_height : ℝ)
  (num_bricks : ℕ)
  (h_wall_length : wall_length = 29)
  (h_wall_width : wall_width = 2)
  (h_wall_height : wall_height = 0.75)
  (h_brick_length : brick_length = 20)
  (h_brick_height : brick_height = 7.5)
  (h_num_bricks : num_bricks = 29000)
  : ∃ (brick_width : ℝ), 
    wall_length * wall_width * wall_height * 1000000 = 
    num_bricks * brick_length * brick_width * brick_height ∧ 
    brick_width = 10 := by
  sorry

end brick_width_is_10cm_l2826_282637


namespace euclids_lemma_l2826_282683

theorem euclids_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) : p ∣ a ∨ p ∣ b := by
  sorry

-- Gauss's lemma (given)
axiom gauss_lemma (p a b : ℕ) (hp : Prime p) (hab : p ∣ a * b) (hna : ¬(p ∣ a)) : p ∣ b

end euclids_lemma_l2826_282683


namespace height_comparison_equivalences_l2826_282634

-- Define the classes A and B
variable (A B : Type)

-- Define a height function for students
variable (height : A ⊕ B → ℝ)

-- Define the propositions for each question
def tallest_A_taller_than_tallest_B : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def every_B_shorter_than_some_A : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def for_any_A_exists_shorter_B : Prop :=
  ∀ a : A, ∃ b : B, height (Sum.inl a) > height (Sum.inr b)

def shortest_B_shorter_than_shortest_A : Prop :=
  ∃ a : A, ∀ b : B, height (Sum.inl a) > height (Sum.inr b)

-- State the theorem
theorem height_comparison_equivalences
  (A B : Type) (height : A ⊕ B → ℝ) :
  (tallest_A_taller_than_tallest_B A B height ↔ every_B_shorter_than_some_A A B height) ∧
  (for_any_A_exists_shorter_B A B height ↔ shortest_B_shorter_than_shortest_A A B height) :=
sorry

end height_comparison_equivalences_l2826_282634


namespace smallest_square_area_for_two_rectangles_l2826_282669

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSideLength (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- Theorem: The smallest square area to fit 2×4 and 3×5 rectangles is 25 -/
theorem smallest_square_area_for_two_rectangles :
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  (minSquareSideLength r1 r2)^2 = 25 := by
  sorry

#eval (minSquareSideLength ⟨2, 4⟩ ⟨3, 5⟩)^2

end smallest_square_area_for_two_rectangles_l2826_282669


namespace discovery_uses_visualization_vr_l2826_282673

/-- Represents different digital Earth technologies -/
inductive DigitalEarthTechnology
  | InformationSuperhighway
  | HighResolutionSatellite
  | SpatialInformation
  | VisualizationAndVirtualReality

/-- Represents a TV program -/
structure TVProgram where
  name : String
  episode : String
  content : String

/-- Determines the digital Earth technology used in a TV program -/
def technology_used (program : TVProgram) : DigitalEarthTechnology :=
  if program.content = "vividly recreated various dinosaurs and their living environments"
  then DigitalEarthTechnology.VisualizationAndVirtualReality
  else DigitalEarthTechnology.InformationSuperhighway

/-- The CCTV Discovery program -/
def discovery_program : TVProgram :=
  { name := "Discovery"
  , episode := "Back to the Dinosaur Era"
  , content := "vividly recreated various dinosaurs and their living environments" }

theorem discovery_uses_visualization_vr :
  technology_used discovery_program = DigitalEarthTechnology.VisualizationAndVirtualReality := by
  sorry


end discovery_uses_visualization_vr_l2826_282673


namespace sum_sequence_37th_term_l2826_282629

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_sequence_37th_term
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (ha1 : a 1 = 25)
  (hb1 : b 1 = 75)
  (hab2 : a 2 + b 2 = 100) :
  a 37 + b 37 = 100 := by
sorry

end sum_sequence_37th_term_l2826_282629


namespace second_discount_percentage_second_discount_is_25_percent_l2826_282662

theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  let second_discount_percent := (second_discount_amount / price_after_first_discount) * 100
  second_discount_percent

theorem second_discount_is_25_percent :
  second_discount_percentage 33.78 25 19 = 25 := by
  sorry

end second_discount_percentage_second_discount_is_25_percent_l2826_282662


namespace three_color_theorem_min_three_colors_min_colors_is_three_l2826_282643

/-- Represents a 3D coordinate in the 3x3x3 grid --/
structure Coord where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a coloring of the 3x3x3 grid --/
def Coloring := Coord → Fin 3

/-- Two coordinates are adjacent if they differ by 1 in exactly one dimension --/
def adjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val + 1 = c2.z.val) ∨
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val = c2.z.val + 1) ∨
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val ∧ c1.z = c2.z) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1 ∧ c1.z = c2.z) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y ∧ c1.z = c2.z) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

/-- A coloring is valid if no adjacent cubes have the same color --/
def validColoring (c : Coloring) : Prop :=
  ∀ c1 c2 : Coord, adjacent c1 c2 → c c1 ≠ c c2

/-- There exists a valid coloring using only 3 colors --/
theorem three_color_theorem : ∃ c : Coloring, validColoring c :=
  sorry

/-- Any valid coloring must use at least 3 colors --/
theorem min_three_colors (c : Coloring) (h : validColoring c) :
  ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3 :=
  sorry

/-- The minimum number of colors needed is exactly 3 --/
theorem min_colors_is_three :
  (∃ c : Coloring, validColoring c) ∧
  (∀ c : Coloring, validColoring c →
    ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3) :=
  sorry

end three_color_theorem_min_three_colors_min_colors_is_three_l2826_282643


namespace equal_intercept_line_equation_l2826_282654

/-- A line with equal intercepts on both axes passing through (2, -3) -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, -3) -/
  point_condition : -3 = m * 2 + b
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = -b / m

/-- The equation of a line with equal intercepts passing through (2, -3) is either x + y + 1 = 0 or 3x + 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 1) ∨ (l.m = -3/2 ∧ l.b = 0) := by
  sorry

end equal_intercept_line_equation_l2826_282654


namespace linda_original_correct_l2826_282630

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

/-- Linda's original amount of money -/
def linda_original : ℕ := 10

/-- Theorem stating that Linda's original amount is correct -/
theorem linda_original_correct : 
  lucy_original - transfer_amount = linda_original + transfer_amount := by
  sorry

end linda_original_correct_l2826_282630


namespace bricks_to_fill_road_l2826_282600

/-- Calculates the number of bricks needed to fill a rectangular road without overlapping -/
theorem bricks_to_fill_road (road_width road_length brick_width brick_height : ℝ) :
  road_width = 6 →
  road_length = 4 →
  brick_width = 0.6 →
  brick_height = 0.2 →
  (road_width * road_length) / (brick_width * brick_height) = 200 := by
  sorry

end bricks_to_fill_road_l2826_282600


namespace probability_is_one_over_930_l2826_282640

/-- Represents a sequence of 40 distinct real numbers -/
def Sequence := { s : Fin 40 → ℝ // Function.Injective s }

/-- The operation that compares and potentially swaps adjacent elements -/
def operation (s : Sequence) : Sequence := sorry

/-- The probability that the 20th element moves to the 30th position after one operation -/
def probability_20_to_30 (s : Sequence) : ℚ := sorry

/-- Theorem stating that the probability is 1/930 -/
theorem probability_is_one_over_930 (s : Sequence) : 
  probability_20_to_30 s = 1 / 930 := by sorry

end probability_is_one_over_930_l2826_282640


namespace four_fives_equal_100_l2826_282658

/-- An arithmetic expression using fives -/
inductive FiveExpr
  | Const : FiveExpr
  | Add : FiveExpr → FiveExpr → FiveExpr
  | Sub : FiveExpr → FiveExpr → FiveExpr
  | Mul : FiveExpr → FiveExpr → FiveExpr

/-- Evaluate a FiveExpr to an integer -/
def eval : FiveExpr → Int
  | FiveExpr.Const => 5
  | FiveExpr.Add a b => eval a + eval b
  | FiveExpr.Sub a b => eval a - eval b
  | FiveExpr.Mul a b => eval a * eval b

/-- Count the number of fives in a FiveExpr -/
def countFives : FiveExpr → Nat
  | FiveExpr.Const => 1
  | FiveExpr.Add a b => countFives a + countFives b
  | FiveExpr.Sub a b => countFives a + countFives b
  | FiveExpr.Mul a b => countFives a + countFives b

/-- Theorem: There exists an arithmetic expression using exactly four fives that equals 100 -/
theorem four_fives_equal_100 : ∃ e : FiveExpr, countFives e = 4 ∧ eval e = 100 := by
  sorry


end four_fives_equal_100_l2826_282658


namespace line_equation_from_parabola_intersections_l2826_282671

/-- Given a parabola y^2 = 2x and a point G, prove that the line AB formed by
    the intersection of two lines from G to the parabola has a specific equation. -/
theorem line_equation_from_parabola_intersections
  (G : ℝ × ℝ)
  (k₁ k₂ : ℝ)
  (h_G : G = (2, 2))
  (h_parabola : ∀ x y, y^2 = 2*x → (∃ A B : ℝ × ℝ, 
    (A.1 = x ∧ A.2 = y) ∨ (B.1 = x ∧ B.2 = y)))
  (h_slopes : ∀ A B : ℝ × ℝ, 
    (A.2^2 = 2*A.1 ∧ B.2^2 = 2*B.1) → 
    k₁ = (A.2 - G.2) / (A.1 - G.1) ∧
    k₂ = (B.2 - G.2) / (B.1 - G.1))
  (h_sum : k₁ + k₂ = 5)
  (h_product : k₁ * k₂ = -2) :
  ∃ A B : ℝ × ℝ, 2 * A.1 + 9 * A.2 + 12 = 0 ∧
                 2 * B.1 + 9 * B.2 + 12 = 0 :=
by sorry

end line_equation_from_parabola_intersections_l2826_282671


namespace complex_fraction_equality_l2826_282695

theorem complex_fraction_equality : (10 * Complex.I) / (2 - Complex.I) = -2 + 4 * Complex.I := by
  sorry

end complex_fraction_equality_l2826_282695


namespace percentage_increase_l2826_282659

theorem percentage_increase (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 78) : 
  (new_earnings - original_earnings) / original_earnings * 100 = 30 := by
  sorry

end percentage_increase_l2826_282659


namespace consecutive_sets_summing_to_150_l2826_282694

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : start * length + (length * (length - 1)) / 2 = 150
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly 3 sets of consecutive positive integers summing to 150 -/
theorem consecutive_sets_summing_to_150 : 
  ∃! (sets : Finset ConsecutiveSet), sets.card = 3 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 150) ∧
    (∀ a b : ℕ, a > 0 → b ≥ 2 → 
      (a * b + (b * (b - 1)) / 2 = 150 → ∃ s ∈ sets, s.start = a ∧ s.length = b)) :=
sorry

end consecutive_sets_summing_to_150_l2826_282694


namespace temple_shop_cost_l2826_282652

/-- The cost per object at the shop --/
def cost_per_object : ℕ := 11

/-- The number of people in Nathan's group --/
def number_of_people : ℕ := 3

/-- The number of shoes per person --/
def shoes_per_person : ℕ := 2

/-- The number of socks per person --/
def socks_per_person : ℕ := 2

/-- The number of mobiles per person --/
def mobiles_per_person : ℕ := 1

/-- The total cost for Nathan and his parents to store their belongings --/
def total_cost : ℕ := number_of_people * (shoes_per_person + socks_per_person + mobiles_per_person) * cost_per_object

theorem temple_shop_cost : total_cost = 165 := by
  sorry

end temple_shop_cost_l2826_282652


namespace square_difference_l2826_282625

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end square_difference_l2826_282625


namespace joel_stuffed_animals_l2826_282636

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := 13

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of toys that were Joel's own -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister gave him -/
def sisters_toys : ℕ := (joels_toys / 2)

theorem joel_stuffed_animals :
  stuffed_animals + action_figures + board_games + puzzles + sisters_toys + joels_toys = total_toys :=
by sorry

end joel_stuffed_animals_l2826_282636


namespace symmetric_difference_A_B_l2826_282690

-- Define the set difference operation
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  set_difference M N ∪ set_difference N M

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -(x-1)^2 + 2}

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {y | y ≤ 0 ∨ y > 2} := by sorry

end symmetric_difference_A_B_l2826_282690


namespace odd_function_properties_l2826_282685

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x ↦ 2^x

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ (1 - h x) / (1 + h x)

-- State the theorem
theorem odd_function_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (h 2 = 4) ∧             -- h(2) = 4
  (∀ x, f x = (1 - 2^x) / (1 + 2^x)) ∧  -- Analytical form of f
  (∀ x, f (2*x - 1) > f (x + 1) ↔ x < 2/3) := by
  sorry

end odd_function_properties_l2826_282685


namespace theater_ticket_pricing_l2826_282605

theorem theater_ticket_pricing (adult_price : ℝ) 
  (h1 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (0.75 * adult_price) = 35) :
  10 * adult_price + 8 * (adult_price / 2) + 5 * (0.75 * adult_price) = 88.75 := by
  sorry

end theater_ticket_pricing_l2826_282605


namespace jaylen_kristin_bell_pepper_ratio_l2826_282612

/-- Prove that the ratio of Jaylen's bell peppers to Kristin's bell peppers is 2:1 -/
theorem jaylen_kristin_bell_pepper_ratio :
  let jaylen_carrots : ℕ := 5
  let jaylen_cucumbers : ℕ := 2
  let kristin_bell_peppers : ℕ := 2
  let kristin_green_beans : ℕ := 20
  let jaylen_green_beans : ℕ := kristin_green_beans / 2 - 3
  let jaylen_total_vegetables : ℕ := 18
  let jaylen_bell_peppers : ℕ := jaylen_total_vegetables - (jaylen_carrots + jaylen_cucumbers + jaylen_green_beans)
  
  (jaylen_bell_peppers : ℚ) / kristin_bell_peppers = 2 := by
  sorry


end jaylen_kristin_bell_pepper_ratio_l2826_282612


namespace cookie_baking_problem_l2826_282680

theorem cookie_baking_problem (x : ℚ) : 
  x > 0 → 
  x + x/2 + (3*x/2 - 4) = 92 → 
  x = 32 := by
sorry

end cookie_baking_problem_l2826_282680


namespace investment_proof_l2826_282678

/-- Represents the total amount invested -/
def total_investment : ℝ := 10000

/-- Represents the amount invested at 6% interest -/
def investment_at_6_percent : ℝ := 7200

/-- Represents the annual interest rate for the first part of the investment -/
def interest_rate_1 : ℝ := 0.06

/-- Represents the annual interest rate for the second part of the investment -/
def interest_rate_2 : ℝ := 0.09

/-- Represents the total interest received after one year -/
def total_interest : ℝ := 684

theorem investment_proof : 
  interest_rate_1 * investment_at_6_percent + 
  interest_rate_2 * (total_investment - investment_at_6_percent) = 
  total_interest :=
by sorry

end investment_proof_l2826_282678


namespace a_plus_b_value_l2826_282604

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 4 * x + 3) → a + b = 13 / 3 := by
  sorry

end a_plus_b_value_l2826_282604


namespace quadratic_factorization_l2826_282682

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end quadratic_factorization_l2826_282682


namespace remainder_problem_l2826_282699

theorem remainder_problem (k : ℕ+) (h : 120 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 2 := by
  sorry

end remainder_problem_l2826_282699


namespace inscribed_circle_circumference_l2826_282688

/-- Given a circle with radius R and an arc subtending 120°, 
    the radius r of the circle inscribed between this arc and its tangents 
    satisfies 2πr = (2πR)/3 -/
theorem inscribed_circle_circumference (R r : ℝ) : r = R / 3 → 2 * π * r = 2 * π * R / 3 := by
  sorry

end inscribed_circle_circumference_l2826_282688


namespace james_training_sessions_l2826_282631

/-- James' training schedule -/
structure TrainingSchedule where
  hoursPerSession : ℕ
  daysOffPerWeek : ℕ
  totalHoursPerYear : ℕ

/-- Calculate the number of training sessions per day -/
def sessionsPerDay (schedule : TrainingSchedule) : ℚ :=
  let daysPerWeek : ℕ := 7
  let weeksPerYear : ℕ := 52
  let trainingDaysPerYear : ℕ := (daysPerWeek - schedule.daysOffPerWeek) * weeksPerYear
  let hoursPerDay : ℚ := schedule.totalHoursPerYear / trainingDaysPerYear
  hoursPerDay / schedule.hoursPerSession

/-- Theorem: James trains 2 times per day -/
theorem james_training_sessions (james : TrainingSchedule) 
  (h1 : james.hoursPerSession = 4)
  (h2 : james.daysOffPerWeek = 2)
  (h3 : james.totalHoursPerYear = 2080) : 
  sessionsPerDay james = 2 := by
  sorry


end james_training_sessions_l2826_282631


namespace double_reflection_of_H_l2826_282667

-- Define the point type
def Point := ℝ × ℝ

-- Define the parallelogram
def E : Point := (3, 6)
def F : Point := (5, 10)
def G : Point := (7, 6)
def H : Point := (5, 2)

-- Define reflection across x-axis
def reflect_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Define reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : Point) : Point :=
  (p.2 - 2, p.1 + 2)

-- Define the composition of the two reflections
def double_reflection (p : Point) : Point :=
  reflect_y_eq_x_plus_2 (reflect_x_axis p)

-- Theorem statement
theorem double_reflection_of_H :
  double_reflection H = (-4, 7) := by sorry

end double_reflection_of_H_l2826_282667


namespace quadratic_translation_l2826_282608

/-- Given a quadratic function f(x) = 2x^2, translating its graph upwards by 2 units
    results in the function g(x) = 2x^2 + 2. -/
theorem quadratic_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2
  let g : ℝ → ℝ := λ x => 2 * x^2 + 2
  g x = f x + 2 := by sorry

end quadratic_translation_l2826_282608


namespace no_solution_lcm_gcd_equation_l2826_282639

theorem no_solution_lcm_gcd_equation : ¬ ∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 300 := by
  sorry

end no_solution_lcm_gcd_equation_l2826_282639


namespace problem_1_problem_2_problem_3_l2826_282606

-- Problem 1
theorem problem_1 (x y : ℝ) : (-4 * x * y^3) * (-2 * x)^2 = -16 * x^3 * y^3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3*x - 2) * (2*x - 3) - (x - 1) * (6*x + 5) = -12*x + 11 := by sorry

-- Problem 3
theorem problem_3 : (3 * (10^2)) * (5 * (10^5)) = (1.5 : ℝ) * (10^8) := by sorry

end problem_1_problem_2_problem_3_l2826_282606


namespace max_product_constraint_l2826_282686

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = 1) :
  a * b ≤ 1 / 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 1 ∧ a₀ * b₀ = 1 / 16 :=
sorry

end max_product_constraint_l2826_282686


namespace smallest_floor_sum_l2826_282649

theorem smallest_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋ ≥ 7 := by
  sorry

end smallest_floor_sum_l2826_282649


namespace num_divisors_10_factorial_l2826_282657

/-- The number of positive divisors of n! -/
def numDivisorsFactorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 10! is 192 -/
theorem num_divisors_10_factorial :
  numDivisorsFactorial 10 = 192 := by sorry

end num_divisors_10_factorial_l2826_282657


namespace inequality_solution_set_function_domain_set_l2826_282670

-- Part 1: Inequality solution
def inequality_solution (x : ℝ) : Prop :=
  x * (x + 2) > x * (3 - x) + 6

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_solution x ↔ (x < -3/2 ∨ x > 2) :=
sorry

-- Part 2: Function domain
def function_domain (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 1 ∧ -x^2 - x + 6 > 0

theorem function_domain_set :
  ∀ x : ℝ, function_domain x ↔ (-1 ≤ x ∧ x < 2 ∧ x ≠ 1) :=
sorry

end inequality_solution_set_function_domain_set_l2826_282670


namespace shortest_to_longest_diagonal_ratio_l2826_282693

/-- A regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon is 1/2 -/
theorem shortest_to_longest_diagonal_ratio (o : RegularOctagon) :
  shortest_diagonal o / longest_diagonal o = 1 / 2 :=
sorry

end shortest_to_longest_diagonal_ratio_l2826_282693


namespace no_integer_solution_x2_plus_y2_eq_3z2_l2826_282603

theorem no_integer_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end no_integer_solution_x2_plus_y2_eq_3z2_l2826_282603


namespace water_remaining_in_bucket_l2826_282691

theorem water_remaining_in_bucket (initial_water : ℚ) (poured_out : ℚ) : 
  initial_water = 3/4 → poured_out = 1/3 → initial_water - poured_out = 5/12 := by
  sorry

end water_remaining_in_bucket_l2826_282691


namespace apple_tree_production_decrease_l2826_282696

theorem apple_tree_production_decrease (season1 season2 season3 total : ℕ) : 
  season1 = 200 →
  season3 = 2 * season2 →
  total = season1 + season2 + season3 →
  total = 680 →
  (season1 - season2 : ℚ) / season1 = 1/5 := by sorry

end apple_tree_production_decrease_l2826_282696


namespace initial_water_percentage_l2826_282633

theorem initial_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 36 →
  final_fraction = 3/4 →
  ∃ initial_percentage : ℝ,
    initial_percentage = 30 ∧
    (initial_percentage / 100) * capacity + added_water = final_fraction * capacity :=
by sorry

end initial_water_percentage_l2826_282633


namespace lcm_problem_l2826_282674

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 50 = 200) : 
  m = 10 := by
  sorry

end lcm_problem_l2826_282674


namespace cats_left_after_sale_l2826_282698

/-- Calculates the number of cats left after a sale --/
theorem cats_left_after_sale (siamese house persian maine_coon : ℕ)
  (siamese_sold house_sold persian_sold maine_coon_sold : ℚ)
  (h_siamese : siamese = 38)
  (h_house : house = 25)
  (h_persian : persian = 15)
  (h_maine_coon : maine_coon = 12)
  (h_siamese_sold : siamese_sold = 60 / 100)
  (h_house_sold : house_sold = 40 / 100)
  (h_persian_sold : persian_sold = 75 / 100)
  (h_maine_coon_sold : maine_coon_sold = 50 / 100) :
  ⌊siamese - siamese * siamese_sold⌋ +
  ⌊house - house * house_sold⌋ +
  ⌊persian - persian * persian_sold⌋ +
  ⌊maine_coon - maine_coon * maine_coon_sold⌋ = 41 := by
  sorry


end cats_left_after_sale_l2826_282698


namespace a_neg_one_necessary_not_sufficient_l2826_282624

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 2 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem a_neg_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = -1) ∧ 
  ¬(∀ a : ℝ, a = -1 → parallel a) :=
sorry

end a_neg_one_necessary_not_sufficient_l2826_282624


namespace unique_k_value_l2826_282664

/-- A predicate to check if a number is a non-zero digit -/
def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The expression as a function of k and t -/
def expression (k t : ℕ) : ℤ := 8 * k * 100 + 8 + k * 100 + 88 - 16 * t * 10 - 6

theorem unique_k_value :
  ∀ k t : ℕ,
  is_nonzero_digit k →
  is_nonzero_digit t →
  t = 6 →
  (∃ m : ℤ, expression k t = m) →
  k = 9 := by sorry

end unique_k_value_l2826_282664


namespace birth_ticket_cost_l2826_282617

/-- The cost of a ticket to Mars at a given time -/
def ticket_cost (years_since_birth : ℕ) : ℚ := sorry

/-- The cost is halved every 10 years -/
axiom cost_halves (y : ℕ) : ticket_cost (y + 10) = ticket_cost y / 2

/-- When Matty is 30, a ticket costs $125,000 -/
axiom cost_at_30 : ticket_cost 30 = 125000

/-- The cost of a ticket to Mars when Matty was born was $1,000,000 -/
theorem birth_ticket_cost : ticket_cost 0 = 1000000 := by sorry

end birth_ticket_cost_l2826_282617


namespace divisibility_theorem_l2826_282621

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ ((a+1)^b - 1) := by
  sorry

end divisibility_theorem_l2826_282621


namespace max_sum_diff_unit_vectors_l2826_282646

theorem max_sum_diff_unit_vectors (a b : EuclideanSpace ℝ (Fin 2)) :
  ‖a‖ = 1 → ‖b‖ = 1 → ‖a + b‖ + ‖a - b‖ ≤ 2 * Real.sqrt 2 := by
  sorry

end max_sum_diff_unit_vectors_l2826_282646


namespace james_old_wage_l2826_282663

/-- Jame's old hourly wage -/
def old_wage : ℝ := 16

/-- Jame's new hourly wage -/
def new_wage : ℝ := 20

/-- Jame's old weekly work hours -/
def old_hours : ℝ := 25

/-- Jame's new weekly work hours -/
def new_hours : ℝ := 40

/-- Number of weeks worked per year -/
def weeks_per_year : ℝ := 52

/-- Difference in annual earnings between new and old job -/
def annual_difference : ℝ := 20800

theorem james_old_wage :
  old_wage * old_hours * weeks_per_year + annual_difference = new_wage * new_hours * weeks_per_year :=
by sorry

end james_old_wage_l2826_282663


namespace solution_set_inequality_l2826_282623

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end solution_set_inequality_l2826_282623


namespace expected_winnings_l2826_282684

/-- Represents the outcome of rolling the die -/
inductive DieOutcome
  | Six
  | Odd
  | Even

/-- The probability of rolling a 6 -/
def prob_six : ℚ := 1/4

/-- The probability of rolling an odd number (1, 3, or 5) -/
def prob_odd : ℚ := (1 - prob_six) * (3/5)

/-- The probability of rolling an even number (2 or 4) -/
def prob_even : ℚ := (1 - prob_six) * (2/5)

/-- The payoff for each outcome -/
def payoff (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Six => -2
  | DieOutcome.Odd => 2
  | DieOutcome.Even => 4

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  prob_six * payoff DieOutcome.Six +
  prob_odd * payoff DieOutcome.Odd +
  prob_even * payoff DieOutcome.Even

theorem expected_winnings :
  expected_value = 8/5 := by sorry

end expected_winnings_l2826_282684


namespace three_lines_equidistant_l2826_282666

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Distance between a point and a line --/
def distance_point_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

theorem three_lines_equidistant (A B : ℝ × ℝ) (h : dist A B = 5) :
  ∃! (s : Finset Line), s.card = 3 ∧ 
    (∀ l ∈ s, distance_point_line A l = 2 ∧ distance_point_line B l = 3) :=
sorry

end three_lines_equidistant_l2826_282666


namespace divisibility_by_six_l2826_282627

theorem divisibility_by_six (a x : ℤ) : 
  (∃ k : ℤ, a * (x^3 + a^2 * x^2 + a^2 - 1) = 6 * k) ↔ 
  (∃ t : ℤ, x = 3 * t ∨ x = 3 * t - a^2) :=
by sorry

end divisibility_by_six_l2826_282627


namespace remainder_of_expression_l2826_282644

theorem remainder_of_expression (n : ℕ) : (1 - 90)^10 % 88 = 1 := by
  sorry

end remainder_of_expression_l2826_282644


namespace rectangular_solid_edge_sum_l2826_282656

theorem rectangular_solid_edge_sum :
  ∀ (a b c r : ℝ),
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 352 →
    b = a * r →
    c = a * r^2 →
    a = 4 →
    4 * (a + b + c) = 112 := by
  sorry

end rectangular_solid_edge_sum_l2826_282656


namespace cuboid_edge_lengths_l2826_282651

theorem cuboid_edge_lengths :
  ∀ a b c : ℕ,
  (a * b * c + a * b + b * c + c * a + a + b + c = 2000) →
  ({a, b, c} : Finset ℕ) = {28, 22, 2} := by
sorry

end cuboid_edge_lengths_l2826_282651


namespace logarithm_inequality_l2826_282622

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  Real.log (Real.sqrt (a * b)) = (Real.log a + Real.log b) / 2 ∧ 
  Real.log (Real.sqrt (a * b)) < Real.log ((a + b) / 2) ∧
  Real.log ((a + b) / 2) < Real.log ((a^2 + b^2) / 2) / 2 := by
  sorry

end logarithm_inequality_l2826_282622


namespace existence_of_rationals_l2826_282611

theorem existence_of_rationals (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f ((a + b) / 2) := by
  sorry

end existence_of_rationals_l2826_282611


namespace cube_tetrahedron_surface_area_ratio_l2826_282645

/-- A cube with side length s contains a regular tetrahedron with vertices
    (0,0,0), (s,s,0), (s,0,s), and (0,s,s). The ratio of the surface area of
    the cube to the surface area of the tetrahedron is √3. -/
theorem cube_tetrahedron_surface_area_ratio (s : ℝ) (h : s > 0) :
  let cube_vertices : Fin 8 → ℝ × ℝ × ℝ := fun i =>
    ((i : ℕ) % 2 * s, ((i : ℕ) / 2) % 2 * s, ((i : ℕ) / 4) * s)
  let tetra_vertices : Fin 4 → ℝ × ℝ × ℝ := fun i =>
    match i with
    | 0 => (0, 0, 0)
    | 1 => (s, s, 0)
    | 2 => (s, 0, s)
    | 3 => (0, s, s)
  let cube_surface_area := 6 * s^2
  let tetra_surface_area := 2 * Real.sqrt 3 * s^2
  cube_surface_area / tetra_surface_area = Real.sqrt 3 := by
  sorry

#check cube_tetrahedron_surface_area_ratio

end cube_tetrahedron_surface_area_ratio_l2826_282645


namespace min_value_theorem_l2826_282668

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = 1 → 2/a + 5/b ≥ 2/x + 5/y) ∧ 2/x + 5/y = 2 :=
sorry

end min_value_theorem_l2826_282668


namespace calculate_hourly_wage_l2826_282692

/-- Calculates the hourly wage of a worker given their work conditions and pay --/
theorem calculate_hourly_wage (hours_per_week : ℕ) (deduction_per_lateness : ℕ) 
  (lateness_count : ℕ) (pay_after_deductions : ℕ) : 
  hours_per_week = 18 → 
  deduction_per_lateness = 5 → 
  lateness_count = 3 → 
  pay_after_deductions = 525 → 
  (pay_after_deductions + lateness_count * deduction_per_lateness) / hours_per_week = 30 := by
  sorry

end calculate_hourly_wage_l2826_282692


namespace radio_show_music_commercial_ratio_l2826_282618

/-- Represents a segment of a radio show -/
structure Segment where
  total_time : ℕ
  commercial_time : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ := sorry

theorem radio_show_music_commercial_ratio 
  (segment1 : Segment)
  (segment2 : Segment)
  (segment3 : Segment)
  (h1 : segment1.total_time = 56 ∧ segment1.commercial_time = 22)
  (h2 : segment2.total_time = 84 ∧ segment2.commercial_time = 28)
  (h3 : segment3.total_time = 128 ∧ segment3.commercial_time = 34) :
  simplify_ratio 
    ((segment1.total_time - segment1.commercial_time) + 
     (segment2.total_time - segment2.commercial_time) + 
     (segment3.total_time - segment3.commercial_time))
    (segment1.commercial_time + segment2.commercial_time + segment3.commercial_time) = (46, 21) := by
  sorry

end radio_show_music_commercial_ratio_l2826_282618


namespace wrong_value_correction_l2826_282614

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : wrong_value = 130)
  (h4 : correct_mean = 191.4) :
  let initial_sum := n * initial_mean
  let sum_without_wrong := initial_sum - wrong_value
  let correct_sum := n * correct_mean
  correct_sum - sum_without_wrong + wrong_value = 295 := by
sorry

end wrong_value_correction_l2826_282614


namespace rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l2826_282660

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem rabbit_pairs (n : ℕ) : 
  fib n = if n = 0 then 0 
          else if n = 1 then 1 
          else fib (n - 1) + fib (n - 2) := by
  sorry

theorem rabbit_pairs_base_cases :
  fib 1 = 1 ∧ fib 2 = 1 := by
  sorry

theorem rabbit_pairs_recurrence (n : ℕ) (h : n > 2) :
  fib n = fib (n - 1) + fib (n - 2) := by
  sorry

end rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l2826_282660


namespace gold_quarter_value_ratio_l2826_282609

theorem gold_quarter_value_ratio : 
  let melted_value_per_ounce : ℚ := 100
  let quarter_weight : ℚ := 1 / 5
  let spent_value : ℚ := 1 / 4
  (melted_value_per_ounce * quarter_weight) / spent_value = 80 := by
  sorry

end gold_quarter_value_ratio_l2826_282609


namespace unique_solution_for_equation_l2826_282665

theorem unique_solution_for_equation : ∃! (n : ℕ), 
  ∃ (x : ℕ), x > 0 ∧ 
  n = 2^(2*x - 1) - 5*x - 3 ∧
  n = (2^(x-1) - 1) * (2^x + 1) ∧
  n = 2015 := by
  sorry

end unique_solution_for_equation_l2826_282665


namespace even_function_properties_l2826_282620

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_decreasing : is_decreasing_on f (-5) (-2))
  (h_max : ∀ x, -5 ≤ x ∧ x ≤ -2 → f x ≤ 7) :
  is_increasing_on f 2 5 ∧ ∀ x, 2 ≤ x ∧ x ≤ 5 → f x ≤ 7 :=
by sorry

end even_function_properties_l2826_282620


namespace percent_relation_l2826_282655

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.3 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end percent_relation_l2826_282655


namespace percentage_problem_l2826_282650

theorem percentage_problem (x : ℝ) : x * 0.0005 = 6.178 → x = 12356 := by
  sorry

end percentage_problem_l2826_282650
