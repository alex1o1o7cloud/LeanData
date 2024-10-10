import Mathlib

namespace locus_of_centers_l3265_326581

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₃ with equation (x - 3)² + y² = 25 -/
def C₃ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₃ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₃ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₃ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₃ a b r) → 
  12 * a^2 + 16 * b^2 - 36 * a - 81 = 0 := by
  sorry

end locus_of_centers_l3265_326581


namespace cubic_sum_divided_by_quadratic_sum_l3265_326547

theorem cubic_sum_divided_by_quadratic_sum (a b c : ℚ) 
  (ha : a = 7) (hb : b = 5) (hc : c = -2) : 
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 + c^2) = 460 / 43 := by
  sorry

end cubic_sum_divided_by_quadratic_sum_l3265_326547


namespace angle_problem_l3265_326595

theorem angle_problem (x : ℝ) :
  (x > 0) →
  (x - 30 > 0) →
  (2 * x + (x - 30) = 360) →
  (x = 130) := by
sorry

end angle_problem_l3265_326595


namespace right_angled_triangle_l3265_326513

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * c * (Real.sin (A / 2))^2 = c - b →
  C = π / 2 := by
sorry

end right_angled_triangle_l3265_326513


namespace complement_intersection_theorem_l3265_326518

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 3, 6}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ A) ∩ B = {2, 6} := by
  sorry

end complement_intersection_theorem_l3265_326518


namespace smallest_positive_period_of_f_l3265_326500

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → S < T → ¬ is_periodic f S :=
by sorry

end smallest_positive_period_of_f_l3265_326500


namespace blind_cave_scorpion_diet_l3265_326583

/-- The number of segments in the first millipede eaten by a blind cave scorpion -/
def first_millipede_segments : ℕ := 60

/-- The total number of segments the scorpion needs to eat daily -/
def total_required_segments : ℕ := 800

/-- The number of additional 50-segment millipedes the scorpion needs to eat -/
def additional_millipedes : ℕ := 10

/-- The number of segments in each additional millipede -/
def segments_per_additional_millipede : ℕ := 50

theorem blind_cave_scorpion_diet (x : ℕ) :
  x = first_millipede_segments ↔
    x + 2 * (2 * x) + additional_millipedes * segments_per_additional_millipede = total_required_segments :=
by
  sorry

end blind_cave_scorpion_diet_l3265_326583


namespace unique_intersection_l3265_326552

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-1, 1-a, 9}

theorem unique_intersection (a : ℝ) : A a ∩ B a = {9} → a = 3 := by
  sorry

end unique_intersection_l3265_326552


namespace archimedes_academy_students_l3265_326528

/-- The number of distinct students preparing for AMC 8 at Archimedes Academy -/
def distinct_students (algebra_students calculus_students statistics_students overlap : ℕ) : ℕ :=
  algebra_students + calculus_students + statistics_students - overlap

/-- Theorem stating the number of distinct students preparing for AMC 8 at Archimedes Academy -/
theorem archimedes_academy_students :
  distinct_students 13 10 12 3 = 32 := by
  sorry

end archimedes_academy_students_l3265_326528


namespace inequality_implies_absolute_value_order_l3265_326564

theorem inequality_implies_absolute_value_order 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h : a^2 / (b^2 + c^2) < b^2 / (c^2 + a^2) ∧ b^2 / (c^2 + a^2) < c^2 / (a^2 + b^2)) : 
  |a| < |b| ∧ |b| < |c| := by
  sorry

end inequality_implies_absolute_value_order_l3265_326564


namespace biased_coin_expected_value_l3265_326586

/-- A biased coin with probability of heads and tails, and corresponding gains/losses -/
structure BiasedCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ

/-- The expected value of a coin flip -/
def expected_value (c : BiasedCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem: The expected value of the specific biased coin is 0 -/
theorem biased_coin_expected_value :
  let c : BiasedCoin := {
    prob_heads := 2/3,
    prob_tails := 1/3,
    gain_heads := 5,
    loss_tails := 10
  }
  expected_value c = 0 := by sorry

end biased_coin_expected_value_l3265_326586


namespace exactly_one_negative_l3265_326510

theorem exactly_one_negative 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (hy₃ : y₃ ≠ 0) 
  (v₁ : ℝ) (hv₁ : v₁ = x₁ + y₁)
  (v₂ : ℝ) (hv₂ : v₂ = x₂ + y₂)
  (v₃ : ℝ) (hv₃ : v₃ = x₃ + y₃)
  (h_prod : x₁ * x₂ * x₃ = -(y₁ * y₂ * y₃))
  (h_sum_squares : x₁^2 + x₂^2 + x₃^2 = y₁^2 + y₂^2 + y₃^2)
  (h_triangle : v₁ + v₂ ≥ v₃ ∧ v₂ + v₃ ≥ v₁ ∧ v₃ + v₁ ≥ v₂)
  (h_triangle_squares : v₁^2 + v₂^2 ≥ v₃^2 ∧ v₂^2 + v₃^2 ≥ v₁^2 ∧ v₃^2 + v₁^2 ≥ v₂^2) :
  (x₁ < 0 ∨ x₂ < 0 ∨ x₃ < 0 ∨ y₁ < 0 ∨ y₂ < 0 ∨ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ x₂ < 0) ∧ ¬(x₁ < 0 ∧ x₃ < 0) ∧ ¬(x₂ < 0 ∧ x₃ < 0) ∧
  ¬(y₁ < 0 ∧ y₂ < 0) ∧ ¬(y₁ < 0 ∧ y₃ < 0) ∧ ¬(y₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ y₁ < 0) ∧ ¬(x₁ < 0 ∧ y₂ < 0) ∧ ¬(x₁ < 0 ∧ y₃ < 0) ∧
  ¬(x₂ < 0 ∧ y₁ < 0) ∧ ¬(x₂ < 0 ∧ y₂ < 0) ∧ ¬(x₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₃ < 0 ∧ y₁ < 0) ∧ ¬(x₃ < 0 ∧ y₂ < 0) ∧ ¬(x₃ < 0 ∧ y₃ < 0) := by
  sorry

end exactly_one_negative_l3265_326510


namespace randy_blocks_left_l3265_326585

/-- Calculates the number of blocks Randy has left after a series of actions. -/
def blocks_left (initial : ℕ) (used : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given_away + bought

/-- Proves that Randy has 70 blocks left after his actions. -/
theorem randy_blocks_left : 
  blocks_left 78 19 25 36 = 70 := by
  sorry

end randy_blocks_left_l3265_326585


namespace root_sum_ratio_l3265_326597

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 85/2 := by
sorry

end root_sum_ratio_l3265_326597


namespace fraction_addition_l3265_326548

theorem fraction_addition : (8 : ℚ) / 12 + (7 : ℚ) / 15 = (17 : ℚ) / 15 := by
  sorry

end fraction_addition_l3265_326548


namespace consecutive_integers_average_l3265_326511

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 = a + 4) :=
by sorry

end consecutive_integers_average_l3265_326511


namespace jennas_tanning_schedule_l3265_326534

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule :
  ∀ (x : ℝ),
  (x ≥ 0) →  -- Non-negative tanning time
  (4 * x + 80 ≤ 200) →  -- Total tanning time constraint
  (x = 30) :=  -- Prove that x is 30 minutes
by
  sorry

end jennas_tanning_schedule_l3265_326534


namespace pyramid_volume_l3265_326507

/-- A cube ABCDEFGH with volume 8 -/
structure Cube :=
  (volume : ℝ)
  (is_cube : volume = 8)

/-- Pyramid ACDH within the cube ABCDEFGH -/
def pyramid (c : Cube) : ℝ := sorry

/-- Theorem: The volume of pyramid ACDH is 4/3 -/
theorem pyramid_volume (c : Cube) : pyramid c = 4/3 := by
  sorry

end pyramid_volume_l3265_326507


namespace meeting_participants_count_l3265_326584

theorem meeting_participants_count :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 750 :=
by sorry

end meeting_participants_count_l3265_326584


namespace probability_two_red_cards_modified_deck_l3265_326512

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)

/-- The probability of drawing two red cards in succession from the modified deck -/
def probability_two_red_cards (deck : ModifiedDeck) : ℚ :=
  (deck.red_cards * (deck.red_cards - 1)) / (deck.total_cards * (deck.total_cards - 1))

/-- Theorem stating the probability of drawing two red cards from the modified deck -/
theorem probability_two_red_cards_modified_deck :
  ∃ (deck : ModifiedDeck),
    deck.total_cards = 60 ∧
    deck.red_cards = 24 ∧
    deck.suits = 5 ∧
    deck.cards_per_suit = 12 ∧
    probability_two_red_cards deck = 92 / 590 := by
  sorry

end probability_two_red_cards_modified_deck_l3265_326512


namespace function_lower_bound_l3265_326593

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m for all real x,
    if f(x) + 9 ≥ 0 for all real x, then m ≥ 3/2 --/
theorem function_lower_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) → m ≥ 3/2 := by
  sorry

end function_lower_bound_l3265_326593


namespace tower_surface_area_l3265_326538

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the surface area of a cube given its visible faces -/
def surfaceArea (cube : Cube) (visibleFaces : ℕ) : ℕ :=
  visibleFaces * cube.sideLength * cube.sideLength

/-- Represents a tower of cubes -/
def CubeTower := List Cube

/-- Calculates the total surface area of a tower of cubes -/
def totalSurfaceArea (tower : CubeTower) : ℕ :=
  match tower with
  | [] => 0
  | [c] => surfaceArea c 6  -- Top cube has all 6 faces visible
  | c :: rest => surfaceArea c 5 + (rest.map (surfaceArea · 4)).sum

theorem tower_surface_area :
  let tower : CubeTower := [
    { sideLength := 1 },
    { sideLength := 2 },
    { sideLength := 3 },
    { sideLength := 4 },
    { sideLength := 5 },
    { sideLength := 6 },
    { sideLength := 7 }
  ]
  totalSurfaceArea tower = 610 := by sorry

end tower_surface_area_l3265_326538


namespace sum_in_base_nine_l3265_326568

/-- Represents a number in base 9 --/
def BaseNine : Type := List Nat

/-- Converts a base 9 number to its decimal representation --/
def to_decimal (n : BaseNine) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

/-- Theorem: The sum of 254₉, 367₉, and 142₉ is 774₉ in base 9 --/
theorem sum_in_base_nine :
  let a : BaseNine := [4, 5, 2]
  let b : BaseNine := [7, 6, 3]
  let c : BaseNine := [2, 4, 1]
  let result : BaseNine := [4, 7, 7]
  add_base_nine (add_base_nine a b) c = result :=
sorry

end sum_in_base_nine_l3265_326568


namespace trig_identity_l3265_326524

theorem trig_identity (α : ℝ) (h : Real.sin α + 3 * Real.cos α = 0) : 
  2 * Real.sin (2 * α) - (Real.cos α)^2 = -13/10 := by
  sorry

end trig_identity_l3265_326524


namespace total_fudge_eaten_l3265_326543

-- Define the conversion rate from pounds to ounces
def pounds_to_ounces : ℝ → ℝ := (· * 16)

-- Define the amount of fudge eaten by each person in pounds
def tomas_fudge : ℝ := 1.5
def katya_fudge : ℝ := 0.5
def boris_fudge : ℝ := 2

-- Theorem statement
theorem total_fudge_eaten :
  pounds_to_ounces tomas_fudge +
  pounds_to_ounces katya_fudge +
  pounds_to_ounces boris_fudge = 64 := by
  sorry

end total_fudge_eaten_l3265_326543


namespace ceiling_floor_difference_l3265_326563

theorem ceiling_floor_difference (x : ℤ) :
  let y : ℚ := 1/2
  (⌈(x : ℚ) + y⌉ - ⌊(x : ℚ) + y⌋ : ℤ) = 1 ∧ 
  (⌈(x : ℚ) + y⌉ - ((x : ℚ) + y) : ℚ) = 1/2 :=
by sorry

end ceiling_floor_difference_l3265_326563


namespace weekend_rain_probability_l3265_326557

theorem weekend_rain_probability
  (p_saturday : ℝ)
  (p_sunday : ℝ)
  (p_sunday_given_saturday : ℝ)
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : p_sunday_given_saturday = 0.8) :
  1 - ((1 - p_saturday) * (1 - p_sunday) + p_saturday * (1 - p_sunday_given_saturday)) = 0.66 := by
  sorry

end weekend_rain_probability_l3265_326557


namespace inequality_proofs_l3265_326504

theorem inequality_proofs :
  (∀ a b : ℝ, a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end inequality_proofs_l3265_326504


namespace nursery_school_students_l3265_326536

theorem nursery_school_students (T : ℕ) 
  (h1 : T / 8 + T / 4 + T / 3 + 40 + 60 = T) 
  (h2 : T / 8 + T / 4 + T / 3 = 100) : T = 142 := by
  sorry

end nursery_school_students_l3265_326536


namespace number_of_students_in_class_l3265_326550

theorem number_of_students_in_class : 
  ∀ (N : ℕ) (avg_age_all avg_age_5 avg_age_9 last_student_age : ℚ),
    avg_age_all = 15 →
    avg_age_5 = 13 →
    avg_age_9 = 16 →
    last_student_age = 16 →
    N * avg_age_all = 5 * avg_age_5 + 9 * avg_age_9 + last_student_age →
    N = 15 := by
  sorry

end number_of_students_in_class_l3265_326550


namespace polynomial_value_at_one_l3265_326578

theorem polynomial_value_at_one (a b c : ℝ) : 
  (-a - b - c + 1 = 6) → (a + b + c + 1 = -4) := by sorry

end polynomial_value_at_one_l3265_326578


namespace power_zero_minus_pi_l3265_326503

theorem power_zero_minus_pi (x : ℝ) : (x - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end power_zero_minus_pi_l3265_326503


namespace p_sufficient_not_necessary_for_q_l3265_326535

theorem p_sufficient_not_necessary_for_q :
  ∃ (a : ℝ), (a = 1 → abs a = 1) ∧ (abs a = 1 → a = 1 → False) := by
sorry

end p_sufficient_not_necessary_for_q_l3265_326535


namespace art_class_problem_l3265_326501

theorem art_class_problem (total_students : ℕ) (total_artworks : ℕ) 
  (first_half_artworks_per_student : ℕ) :
  total_students = 10 →
  total_artworks = 35 →
  first_half_artworks_per_student = 3 →
  ∃ (second_half_artworks_per_student : ℕ),
    (total_students / 2 * first_half_artworks_per_student) + 
    (total_students / 2 * second_half_artworks_per_student) = total_artworks ∧
    second_half_artworks_per_student = 4 :=
by sorry

end art_class_problem_l3265_326501


namespace basketball_game_scores_l3265_326515

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ diff : ℕ, diff > 0 ∧ b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem about the basketball game -/
theorem basketball_game_scores
  (alpha : TeamScores)
  (beta : TeamScores)
  (h1 : alpha.first = beta.first)  -- Tied at the end of first quarter
  (h2 : isGeometricSequence alpha.first alpha.second alpha.third alpha.fourth)
  (h3 : isArithmeticSequence beta.first beta.second beta.third beta.fourth)
  (h4 : alpha.first + alpha.second + alpha.third + alpha.fourth =
        beta.first + beta.second + beta.third + beta.fourth + 2)  -- Alpha won by 2 points
  (h5 : alpha.first + alpha.second + alpha.third + alpha.fourth +
        beta.first + beta.second + beta.third + beta.fourth < 200)  -- Total score under 200
  : alpha.first + alpha.second + beta.first + beta.second = 30 :=
by sorry


end basketball_game_scores_l3265_326515


namespace min_value_product_l3265_326562

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x / y + y / z + z / x + y / x + z / y + x / z = 10) :
  (x / y + y / z + z / x) * (y / x + z / y + x / z) ≥ 25 := by
  sorry

end min_value_product_l3265_326562


namespace max_value_of_f_l3265_326582

theorem max_value_of_f (α : ℝ) :
  ∃ M : ℝ, M = (Real.sqrt 2 + 1) / 2 ∧
  (∀ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) ≤ M) ∧
  (∃ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) = M) :=
by sorry

end max_value_of_f_l3265_326582


namespace tree_growth_theorem_l3265_326553

/-- The height of a tree after n years, given its initial height and growth factor --/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * growth_factor ^ years

/-- Theorem stating that a tree with initial height h quadrupling every year for 4 years
    reaches 256 feet if and only if h = 1 foot --/
theorem tree_growth_theorem (h : ℝ) : 
  tree_height h 4 4 = 256 ↔ h = 1 := by
  sorry

#check tree_growth_theorem

end tree_growth_theorem_l3265_326553


namespace circle_integer_points_l3265_326544

theorem circle_integer_points 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) :
  ∀ (A B : ℤ × ℤ), A ≠ B →
  ¬(∃ (r : ℝ), r > 0 ∧ 
    ((A.1 - center.1)^2 + (A.2 - center.2)^2 = r^2) ∧
    ((B.1 - center.1)^2 + (B.2 - center.2)^2 = r^2)) :=
by sorry

end circle_integer_points_l3265_326544


namespace abs_ratio_sum_l3265_326576

theorem abs_ratio_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a| / a + |b| / b : ℚ) = 2 ∨ (|a| / a + |b| / b : ℚ) = -2 ∨ (|a| / a + |b| / b : ℚ) = 0 :=
by sorry

end abs_ratio_sum_l3265_326576


namespace kobe_initial_order_proof_l3265_326555

/-- Represents the number of pieces of fried chicken Kobe initially ordered -/
def kobe_initial_order : ℕ := 5

/-- Represents the number of pieces of fried chicken Pau initially ordered -/
def pau_initial_order : ℕ := 2 * kobe_initial_order

/-- Represents the total number of pieces of fried chicken Pau ate -/
def pau_total : ℕ := 20

theorem kobe_initial_order_proof :
  pau_initial_order + pau_initial_order = pau_total :=
by sorry

end kobe_initial_order_proof_l3265_326555


namespace bobs_weight_l3265_326567

theorem bobs_weight (j b : ℝ) 
  (sum_condition : j + b = 180)
  (diff_condition : b - j = b / 2) : 
  b = 120 := by sorry

end bobs_weight_l3265_326567


namespace average_speed_three_lap_run_l3265_326522

/-- Calculates the average speed of a three-lap run given the track length and lap times -/
theorem average_speed_three_lap_run (track_length : ℝ) (first_lap_time second_lap_time third_lap_time : ℝ) :
  track_length = 400 →
  first_lap_time = 70 →
  second_lap_time = 85 →
  third_lap_time = 85 →
  (3 * track_length) / (first_lap_time + second_lap_time + third_lap_time) = 5 := by
  sorry

#check average_speed_three_lap_run

end average_speed_three_lap_run_l3265_326522


namespace problem_statement_l3265_326574

theorem problem_statement :
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℚ), x₁ < 0 ∧ x₂ < 0 ∧ x₃ < 0 ∧ x₄ * x₅ > 0 ∧ x₁ * x₂ * x₃ * x₄ * x₅ < 0) ∧
  (∀ m : ℝ, abs m + m = 0 → m ≤ 0) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ (a < b ∨ a > b)) ∧
  (∀ a : ℝ, 5 - abs (a - 5) ≤ 5) ∧ (∃ a : ℝ, 5 - abs (a - 5) = 5) :=
by sorry

end problem_statement_l3265_326574


namespace james_current_age_l3265_326537

-- Define the ages as natural numbers
def james_age : ℕ := sorry
def john_age : ℕ := sorry
def tim_age : ℕ := sorry

-- State the given conditions
axiom age_difference : john_age = james_age + 12
axiom tim_age_relation : tim_age = 2 * john_age - 5
axiom tim_age_value : tim_age = 79

-- Theorem to prove
theorem james_current_age : james_age = 25 := by sorry

end james_current_age_l3265_326537


namespace vector_properties_l3265_326541

def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (7, 1)

theorem vector_properties :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  angle = 3 * Real.pi / 4 ∧ proj = (-1/2) • b := by sorry

end vector_properties_l3265_326541


namespace beast_sports_meeting_l3265_326577

theorem beast_sports_meeting (total : ℕ) (tigers lions leopards : ℕ) : 
  total = 220 →
  lions = 2 * tigers + 5 →
  leopards = 2 * lions - 5 →
  total = tigers + lions + leopards →
  leopards - tigers = 95 :=
by
  sorry

end beast_sports_meeting_l3265_326577


namespace current_speed_l3265_326590

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem current_speed 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : speed_against_current = 8) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2 :=
by
  sorry

end current_speed_l3265_326590


namespace unique_solution_l3265_326505

/-- Represents the number of photos taken by each person -/
structure PhotoCounts where
  C : ℕ  -- Claire
  L : ℕ  -- Lisa
  R : ℕ  -- Robert
  D : ℕ  -- David
  E : ℕ  -- Emma

/-- Checks if the given photo counts satisfy all the conditions -/
def satisfiesConditions (p : PhotoCounts) : Prop :=
  p.L = 3 * p.C ∧
  p.R = p.C + 10 ∧
  p.D = 2 * p.C - 5 ∧
  p.E = 2 * p.R ∧
  p.L + p.R + p.C + p.D + p.E = 350

/-- The unique solution to the photo counting problem -/
def solution : PhotoCounts :=
  { C := 36, L := 108, R := 46, D := 67, E := 93 }

/-- Theorem stating that the solution is unique and satisfies all conditions -/
theorem unique_solution :
  satisfiesConditions solution ∧
  ∀ p : PhotoCounts, satisfiesConditions p → p = solution :=
by sorry

end unique_solution_l3265_326505


namespace quadratic_real_solutions_range_l3265_326525

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + 4 * x + 1

-- Define the condition for real solutions
def has_real_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

-- Theorem statement
theorem quadratic_real_solutions_range :
  ∀ m : ℝ, has_real_solutions m ↔ m ≤ 7 ∧ m ≠ 3 :=
sorry

end quadratic_real_solutions_range_l3265_326525


namespace chemistry_marks_proof_l3265_326509

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 74
def total_subjects : ℕ := 5

def chemistry_marks : ℕ := 67

theorem chemistry_marks_proof :
  chemistry_marks = total_subjects * average_marks - (english_marks + math_marks + physics_marks + biology_marks) :=
by sorry

end chemistry_marks_proof_l3265_326509


namespace smallestDualPalindromeCorrect_l3265_326539

def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallestDualPalindrome : ℕ := 15

theorem smallestDualPalindromeCorrect :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

end smallestDualPalindromeCorrect_l3265_326539


namespace square_side_increase_percentage_l3265_326530

theorem square_side_increase_percentage (a : ℝ) (x : ℝ) :
  (a > 0) →
  (x > 0) →
  (a * (1 + x / 100) * 1.8)^2 = 2.592 * (a^2 + (a * (1 + x / 100))^2) →
  x = 100 := by sorry

end square_side_increase_percentage_l3265_326530


namespace least_valid_number_l3265_326566

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * m + d ∧ 
    1 ≤ d ∧ d ≤ 9 ∧ 
    m = n / 25

theorem least_valid_number : 
  (∀ k < 3125, ¬(is_valid_number k)) ∧ is_valid_number 3125 :=
sorry

end least_valid_number_l3265_326566


namespace ohara_triple_49_16_l3265_326549

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49,16,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end ohara_triple_49_16_l3265_326549


namespace modular_home_cost_modular_home_cost_proof_l3265_326573

/-- Calculates the total cost of a modular home given specific conditions. -/
theorem modular_home_cost (kitchen_area : ℕ) (kitchen_cost : ℕ) 
  (bathroom_area : ℕ) (bathroom_cost : ℕ) (other_cost_per_sqft : ℕ) 
  (total_area : ℕ) (num_bathrooms : ℕ) : ℕ :=
  let total_module_area := kitchen_area + num_bathrooms * bathroom_area
  let remaining_area := total_area - total_module_area
  kitchen_cost + num_bathrooms * bathroom_cost + remaining_area * other_cost_per_sqft

/-- Proves that the total cost of the specified modular home is $174,000. -/
theorem modular_home_cost_proof : 
  modular_home_cost 400 20000 150 12000 100 2000 2 = 174000 := by
  sorry

end modular_home_cost_modular_home_cost_proof_l3265_326573


namespace gcd_of_three_numbers_l3265_326591

theorem gcd_of_three_numbers :
  Nat.gcd 45321 (Nat.gcd 76543 123456) = 3 := by
  sorry

end gcd_of_three_numbers_l3265_326591


namespace andrew_steps_to_meet_ben_l3265_326559

/-- The distance between Andrew's and Ben's houses in feet -/
def distance : ℝ := 21120

/-- The ratio of Ben's speed to Andrew's speed -/
def speed_ratio : ℝ := 3

/-- The length of Andrew's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Andrew takes before meeting Ben -/
def steps : ℕ := 1760

theorem andrew_steps_to_meet_ben :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end andrew_steps_to_meet_ben_l3265_326559


namespace exists_rank_with_profit_2016_l3265_326569

/-- The profit of a firm given its rank -/
def profit : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 1) => profit n + (n + 1)

/-- The theorem stating that there exists a rank with profit 2016 -/
theorem exists_rank_with_profit_2016 : ∃ n : ℕ, profit n = 2016 := by
  sorry

end exists_rank_with_profit_2016_l3265_326569


namespace trihedral_angle_relations_l3265_326520

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  /-- Plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- Dihedral angles of the trihedral angle -/
  dihedral_angles : Fin 3 → ℝ

/-- Theorem about the relationship between plane angles and dihedral angles in a trihedral angle -/
theorem trihedral_angle_relations (t : TrihedralAngle) :
  (∀ i : Fin 3, t.plane_angles i > Real.pi / 2 → ∀ j : Fin 3, t.dihedral_angles j > Real.pi / 2) ∧
  (∀ i : Fin 3, t.dihedral_angles i < Real.pi / 2 → ∀ j : Fin 3, t.plane_angles j < Real.pi / 2) :=
sorry

end trihedral_angle_relations_l3265_326520


namespace hyperbola_axis_ratio_implies_m_l3265_326531

/-- Represents a hyperbola with equation mx^2 + y^2 = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1

/-- The length of the imaginary axis of the hyperbola -/
def imaginary_axis_length (h : Hyperbola m) : ℝ := sorry

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola m) : ℝ := sorry

/-- 
  Theorem: For a hyperbola with equation mx^2 + y^2 = 1, 
  if the length of the imaginary axis is twice the length of the real axis, 
  then m = -1/4
-/
theorem hyperbola_axis_ratio_implies_m (m : ℝ) (h : Hyperbola m) 
  (axis_ratio : imaginary_axis_length h = 2 * real_axis_length h) : 
  m = -1/4 := by sorry

end hyperbola_axis_ratio_implies_m_l3265_326531


namespace stream_speed_l3265_326546

/-- Proves that the speed of the stream is 4 km/hr, given the boat's speed in still water
    and the time and distance traveled downstream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) :
  boat_speed = 16 →
  time = 3 →
  distance = 60 →
  ∃ (stream_speed : ℝ), stream_speed = 4 ∧ distance = (boat_speed + stream_speed) * time :=
by sorry

end stream_speed_l3265_326546


namespace orange_balls_count_l3265_326558

theorem orange_balls_count (black white : ℕ) (p : ℚ) (orange : ℕ) : 
  black = 7 → 
  white = 6 → 
  p = 38095238095238093 / 100000000000000000 →
  (black : ℚ) / (orange + black + white : ℚ) = p →
  orange = 5 :=
by sorry

end orange_balls_count_l3265_326558


namespace soccer_tournament_equation_l3265_326575

theorem soccer_tournament_equation (x : ℕ) (h : x > 1) : 
  (x.choose 2 = 28) ↔ (x * (x - 1) / 2 = 28) := by sorry

end soccer_tournament_equation_l3265_326575


namespace ten_player_tournament_matches_l3265_326598

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry


end ten_player_tournament_matches_l3265_326598


namespace garden_table_bench_ratio_l3265_326565

theorem garden_table_bench_ratio :
  ∀ (table_cost bench_cost : ℕ),
    bench_cost = 150 →
    table_cost + bench_cost = 450 →
    ∃ (k : ℕ), table_cost = k * bench_cost →
    (table_cost : ℚ) / (bench_cost : ℚ) = 2 / 1 :=
by
  sorry

end garden_table_bench_ratio_l3265_326565


namespace sufficient_not_necessary_l3265_326521

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) := by
sorry

end sufficient_not_necessary_l3265_326521


namespace intersection_point_of_perpendicular_chords_l3265_326508

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line
def line (m b x y : ℝ) : Prop := x = m*y + b

-- Define perpendicularity of two points with respect to the origin
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem intersection_point_of_perpendicular_chords :
  ∀ (m b x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line m b x₁ y₁ →
  line m b x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ (x y : ℝ), line m b x y ∧ x = 2 ∧ y = 0 :=
sorry

end intersection_point_of_perpendicular_chords_l3265_326508


namespace standard_colony_conditions_l3265_326572

/-- Represents the type of culture medium -/
inductive CultureMedium
| Liquid
| Solid

/-- Represents a bacterial colony -/
structure BacterialColony where
  initialBacteria : ℕ
  medium : CultureMedium

/-- Defines what constitutes a standard bacterial colony -/
def isStandardColony (colony : BacterialColony) : Prop :=
  colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid

/-- Theorem stating the conditions for a standard bacterial colony -/
theorem standard_colony_conditions :
  ∀ (colony : BacterialColony),
    isStandardColony colony ↔
      colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid :=
by
  sorry

end standard_colony_conditions_l3265_326572


namespace store_profit_is_33_percent_l3265_326592

/-- Calculates the store's profit percentage given the markups, discount, and shipping cost -/
def store_profit_percentage (first_markup : ℝ) (second_markup : ℝ) (discount : ℝ) (shipping_cost : ℝ) : ℝ :=
  let price_after_first_markup := 1 + first_markup
  let price_after_second_markup := price_after_first_markup + second_markup * price_after_first_markup
  let price_after_discount := price_after_second_markup * (1 - discount)
  let total_cost := 1 + shipping_cost
  price_after_discount - total_cost

/-- Theorem stating that the store's profit is 33% of the original cost price -/
theorem store_profit_is_33_percent :
  store_profit_percentage 0.20 0.25 0.08 0.05 = 0.33 := by
  sorry

end store_profit_is_33_percent_l3265_326592


namespace equation_solutions_l3265_326502

theorem equation_solutions :
  (∃ x : ℝ, 0.4 * x = -1.2 * x + 1.6 ∧ x = 1) ∧
  (∃ y : ℝ, (1/3) * (y + 2) = 1 - (1/6) * (2 * y - 1) ∧ y = 3/4) := by
sorry

end equation_solutions_l3265_326502


namespace square_remainder_mod_16_l3265_326589

theorem square_remainder_mod_16 (n : ℤ) : ∃ k : ℤ, 0 ≤ k ∧ k < 4 ∧ (n^2) % 16 = k^2 := by
  sorry

end square_remainder_mod_16_l3265_326589


namespace sum_of_max_min_M_l3265_326571

/-- The set T of points (x, y) satisfying |x+1| + |y-2| ≤ 3 -/
def T : Set (ℝ × ℝ) := {p | |p.1 + 1| + |p.2 - 2| ≤ 3}

/-- The set M of values x + 2y for (x, y) in T -/
def M : Set ℝ := {z | ∃ p ∈ T, z = p.1 + 2 * p.2}

theorem sum_of_max_min_M : (⨆ z ∈ M, z) + (⨅ z ∈ M, z) = 6 := by
  sorry

end sum_of_max_min_M_l3265_326571


namespace outfit_choices_count_l3265_326570

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 5

/-- The number of shirts available -/
def num_shirts : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfit combinations where all items are the same color -/
def same_color_combinations : ℕ := num_colors

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - same_color_combinations

theorem outfit_choices_count : valid_outfit_choices = 120 := by
  sorry

end outfit_choices_count_l3265_326570


namespace multiply_cube_by_negative_l3265_326594

/-- For any real number y, 2y³ * (-y) = -2y⁴ -/
theorem multiply_cube_by_negative (y : ℝ) : 2 * y^3 * (-y) = -2 * y^4 := by
  sorry

end multiply_cube_by_negative_l3265_326594


namespace C_power_50_l3265_326587

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end C_power_50_l3265_326587


namespace certain_number_proof_l3265_326579

theorem certain_number_proof : ∃ x : ℝ, (1/4 * x + 15 = 27) ∧ (x = 48) := by
  sorry

end certain_number_proof_l3265_326579


namespace billys_age_l3265_326506

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end billys_age_l3265_326506


namespace alcohol_concentration_problem_l3265_326545

theorem alcohol_concentration_problem (vessel1_capacity vessel2_capacity total_liquid final_vessel_capacity : ℝ)
  (vessel2_concentration final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 55 / 100 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 / 100 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 20 / 100 ∧
    initial_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
    final_concentration * final_vessel_capacity :=
by sorry

end alcohol_concentration_problem_l3265_326545


namespace range_of_m_l3265_326517

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ m : ℝ, Real.log ((m^2) / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end range_of_m_l3265_326517


namespace min_value_of_exponential_sum_l3265_326516

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (min : ℝ), min = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ min :=
sorry

end min_value_of_exponential_sum_l3265_326516


namespace remainder_sum_l3265_326519

theorem remainder_sum (c d : ℤ) 
  (hc : c % 90 = 84) 
  (hd : d % 120 = 117) : 
  (c + d) % 30 = 21 := by
sorry

end remainder_sum_l3265_326519


namespace shaded_area_of_square_l3265_326540

theorem shaded_area_of_square (r : ℝ) (h1 : r = 1/4) :
  (∑' n, r^n) * r = 1/3 := by sorry

end shaded_area_of_square_l3265_326540


namespace unique_y_for_diamond_eq_21_l3265_326527

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond_eq_21 : ∃! y : ℝ, diamond 4 y = 21 := by
  sorry

end unique_y_for_diamond_eq_21_l3265_326527


namespace min_value_sqrt_sum_squares_l3265_326542

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
sorry

end min_value_sqrt_sum_squares_l3265_326542


namespace inequality_solution_l3265_326588

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if a > 1 then { x | 1/a < x ∧ x < 1 }
  else if a = 1 then ∅
  else if 0 < a ∧ a < 1 then { x | 1 < x ∧ x < 1/a }
  else { x | x < 1/a ∨ x > 1 }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | (a*x - 1)*(x - 1) < 0 } = solution_set a :=
sorry

end inequality_solution_l3265_326588


namespace candy_problem_l3265_326526

theorem candy_problem (total_candies : ℕ) : 
  (∃ (n : ℕ), n > 10 ∧ total_candies = 3 * (n - 1) + 2) ∧ 
  (∃ (m : ℕ), m < 10 ∧ total_candies = 4 * (m - 1) + 3) →
  total_candies = 35 := by
sorry

end candy_problem_l3265_326526


namespace yonderland_license_plates_l3265_326533

/-- The number of possible letters in each position of a license plate. -/
def numLetters : ℕ := 26

/-- The number of possible digits in each position of a license plate. -/
def numDigits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def numLetterPositions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def numDigitPositions : ℕ := 4

/-- The total number of valid license plates in Yonderland. -/
def totalLicensePlates : ℕ := numLetters ^ numLetterPositions * numDigits ^ numDigitPositions

theorem yonderland_license_plates : totalLicensePlates = 175760000 := by
  sorry

end yonderland_license_plates_l3265_326533


namespace inequality_system_solution_l3265_326554

theorem inequality_system_solution (x : ℝ) :
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 := by
  sorry

end inequality_system_solution_l3265_326554


namespace polynomial_division_theorem_l3265_326523

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - 22*x^3 + 12*x^2 - 16*x + 8 = (x - 3) * (x^4 + 3*x^3 - 13*x^2 - 27*x - 97) + (-211) := by
  sorry

end polynomial_division_theorem_l3265_326523


namespace no_valid_box_dimensions_l3265_326561

theorem no_valid_box_dimensions : ¬∃ (b c : ℤ), b ≤ c ∧ 2 * b * c + 2 * (2 * b + 2 * c + b * c) = 120 := by
  sorry

end no_valid_box_dimensions_l3265_326561


namespace prob_rain_theorem_l3265_326596

/-- The probability of rain on at least one day during a three-day period -/
def prob_rain_at_least_once (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem stating the probability of rain on at least one day is 86% -/
theorem prob_rain_theorem :
  prob_rain_at_least_once 0.3 0.6 0.5 = 0.86 := by
  sorry

#eval prob_rain_at_least_once 0.3 0.6 0.5

end prob_rain_theorem_l3265_326596


namespace intersection_sum_l3265_326532

def M : Set ℝ := {x | |x - 4| + |x - 1| < 5}
def N (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6}

theorem intersection_sum (a b : ℝ) : 
  M ∩ N a = {2, b} → a + b = 7 := by
  sorry

end intersection_sum_l3265_326532


namespace sum_of_four_numbers_l3265_326529

/-- Given two-digit numbers EH, OY, AY, and OH, where EH is four times OY 
    and AY is four times OH, prove that their sum is 150. -/
theorem sum_of_four_numbers (EH OY AY OH : ℕ) : 
  (10 ≤ EH) ∧ (EH < 100) ∧
  (10 ≤ OY) ∧ (OY < 100) ∧
  (10 ≤ AY) ∧ (AY < 100) ∧
  (10 ≤ OH) ∧ (OH < 100) ∧
  (EH = 4 * OY) ∧
  (AY = 4 * OH) →
  EH + OY + AY + OH = 150 := by
  sorry

end sum_of_four_numbers_l3265_326529


namespace unique_sum_of_squares_125_l3265_326580

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
def countWaysToSum (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 125 as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
theorem unique_sum_of_squares_125 : countWaysToSum 125 = 1 := by
  sorry

end unique_sum_of_squares_125_l3265_326580


namespace one_fifth_of_seven_x_plus_three_l3265_326551

theorem one_fifth_of_seven_x_plus_three (x : ℝ) : 
  (1 / 5) * (7 * x + 3) = (7 / 5) * x + 3 / 5 := by
  sorry

end one_fifth_of_seven_x_plus_three_l3265_326551


namespace derivative_f_at_one_l3265_326560

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_f_at_one : 
  deriv f 1 = 2 + Real.exp 1 := by sorry

end derivative_f_at_one_l3265_326560


namespace subtraction_of_fractions_l3265_326556

theorem subtraction_of_fractions : (12 : ℚ) / 30 - 1 / 7 = 9 / 35 := by
  sorry

end subtraction_of_fractions_l3265_326556


namespace product_pricing_and_profit_maximization_l3265_326514

/-- Represents the purchase and selling prices of products A and B -/
structure ProductPrices where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ

/-- Represents the number of units purchased for products A and B -/
structure PurchaseUnits where
  units_A : ℕ
  units_B : ℕ

/-- Calculates the total cost of purchasing given units of products A and B -/
def total_cost (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  prices.purchase_price_A * units.units_A + prices.purchase_price_B * units.units_B

/-- Calculates the total profit from selling given units of products A and B -/
def total_profit (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  (prices.selling_price_A - prices.purchase_price_A) * units.units_A +
  (prices.selling_price_B - prices.purchase_price_B) * units.units_B

theorem product_pricing_and_profit_maximization
  (prices : ProductPrices)
  (h1 : prices.purchase_price_B = 80)
  (h2 : prices.selling_price_A = 300)
  (h3 : prices.selling_price_B = 100)
  (h4 : total_cost prices { units_A := 50, units_B := 25 } = 15000)
  (h5 : ∀ units : PurchaseUnits, units.units_A + units.units_B = 300 → units.units_B ≥ 2 * units.units_A) :
  prices.purchase_price_A = 260 ∧
  ∃ (max_units : PurchaseUnits),
    max_units.units_A + max_units.units_B = 300 ∧
    max_units.units_B ≥ 2 * max_units.units_A ∧
    max_units.units_A = 100 ∧
    max_units.units_B = 200 ∧
    total_profit prices max_units = 8000 ∧
    ∀ (units : PurchaseUnits),
      units.units_A + units.units_B = 300 →
      units.units_B ≥ 2 * units.units_A →
      total_profit prices units ≤ total_profit prices max_units := by
  sorry

end product_pricing_and_profit_maximization_l3265_326514


namespace baxter_peanut_purchase_l3265_326599

/-- Calculates the pounds of peanuts purchased over the minimum -/
def peanuts_over_minimum (cost_per_pound : ℚ) (minimum_pounds : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent / cost_per_pound) - minimum_pounds

theorem baxter_peanut_purchase : 
  let cost_per_pound : ℚ := 3
  let minimum_pounds : ℚ := 15
  let total_spent : ℚ := 105
  peanuts_over_minimum cost_per_pound minimum_pounds total_spent = 20 := by
sorry

end baxter_peanut_purchase_l3265_326599
