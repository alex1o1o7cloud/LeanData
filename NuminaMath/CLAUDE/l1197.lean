import Mathlib

namespace NUMINAMATH_CALUDE_no_real_solutions_l1197_119798

theorem no_real_solutions : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1197_119798


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1197_119781

theorem largest_perfect_square_factor_of_1800 : 
  (∃ (n : ℕ), n^2 = 900 ∧ n^2 ∣ 1800 ∧ ∀ (m : ℕ), m^2 ∣ 1800 → m^2 ≤ 900) := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1197_119781


namespace NUMINAMATH_CALUDE_slide_problem_l1197_119791

/-- The number of additional boys who went down the slide -/
def additional_boys (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem slide_problem (initial : ℕ) (total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) : 
  additional_boys initial total = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_slide_problem_l1197_119791


namespace NUMINAMATH_CALUDE_percentage_problem_l1197_119775

theorem percentage_problem (percentage : ℝ) : 
  (percentage * 100 - 20 = 60) → percentage = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1197_119775


namespace NUMINAMATH_CALUDE_equation_solution_l1197_119745

theorem equation_solution (n k l m : ℕ) : 
  l > 1 → 
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1197_119745


namespace NUMINAMATH_CALUDE_parabola_translation_l1197_119755

/-- The initial parabola function -/
def initial_parabola (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The final parabola function -/
def final_parabola (x : ℝ) : ℝ := -3 * x^2

/-- Translation function that moves a point 1 unit right and 2 units up -/
def translate (x y : ℝ) : ℝ × ℝ := (x - 1, y + 2)

theorem parabola_translation :
  ∀ x : ℝ, final_parabola x = (initial_parabola (x - 1) + 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1197_119755


namespace NUMINAMATH_CALUDE_min_value_m_l1197_119769

theorem min_value_m (m : ℝ) (h1 : m > 0)
  (h2 : ∀ x : ℝ, x > 1 → 2 * Real.exp (2 * m * x) - Real.log x / m ≥ 0) :
  m ≥ 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_m_l1197_119769


namespace NUMINAMATH_CALUDE_compound_inequality_solution_l1197_119792

theorem compound_inequality_solution (x : ℝ) :
  (x / 2 ≤ 5 - x ∧ 5 - x < -(3 * (2 + x))) ↔ x < -11/2 := by sorry

end NUMINAMATH_CALUDE_compound_inequality_solution_l1197_119792


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1197_119730

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1197_119730


namespace NUMINAMATH_CALUDE_elijah_coffee_pints_l1197_119735

-- Define the conversion rate from cups to pints
def cups_to_pints : ℚ := 1 / 2

-- Define the total amount of liquid consumed in cups
def total_liquid_cups : ℚ := 36

-- Define the amount of water Emilio drank in pints
def emilio_water_pints : ℚ := 9.5

-- Theorem statement
theorem elijah_coffee_pints :
  (total_liquid_cups * cups_to_pints) - emilio_water_pints = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_elijah_coffee_pints_l1197_119735


namespace NUMINAMATH_CALUDE_streaming_service_fee_l1197_119738

/-- Given a fixed monthly fee and a charge per hour for extra content,
    if the total for one month is $18.60 and the total for another month
    with triple the extra content usage is $32.40,
    then the fixed monthly fee is $11.70. -/
theorem streaming_service_fee (x y : ℝ)
  (feb_bill : x + y = 18.60)
  (mar_bill : x + 3*y = 32.40) :
  x = 11.70 := by
  sorry

end NUMINAMATH_CALUDE_streaming_service_fee_l1197_119738


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1197_119741

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1197_119741


namespace NUMINAMATH_CALUDE_simplify_expression_l1197_119711

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1197_119711


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1197_119754

theorem quadratic_equation_properties (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 9 * x₁ + c = 0 ∧ 3 * x₂^2 - 9 * x₂ + c = 0) →
  (c < 6.75 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1197_119754


namespace NUMINAMATH_CALUDE_tan_beta_value_l1197_119722

theorem tan_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 4/3) (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1197_119722


namespace NUMINAMATH_CALUDE_min_sum_squares_l1197_119799

def S : Set Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
              f ≠ g ∧ f ≠ h ∧
              g ≠ h)
  (in_set : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S)
  (sum_condition : e + f + g + h = 9) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1197_119799


namespace NUMINAMATH_CALUDE_power_of_power_l1197_119763

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1197_119763


namespace NUMINAMATH_CALUDE_total_jelly_beans_l1197_119756

theorem total_jelly_beans (vanilla : ℕ) (grape : ℕ) : 
  vanilla = 120 → 
  grape = 5 * vanilla + 50 → 
  vanilla + grape = 770 := by
sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l1197_119756


namespace NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l1197_119705

/-- The time Nadine spends cleaning her dog -/
def clean_dog_time (hosing_time number_of_shampoos time_per_shampoo : ℕ) : ℕ :=
  hosing_time + number_of_shampoos * time_per_shampoo

/-- Theorem stating that Nadine spends 55 minutes cleaning her dog -/
theorem nadine_dog_cleaning_time :
  clean_dog_time 10 3 15 = 55 :=
by sorry

end NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l1197_119705


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l1197_119761

def total_team_members : ℕ := 20
def starting_lineup : ℕ := 7
def regular_players : ℕ := 5

def choose_team : ℕ := 
  total_team_members * (total_team_members - 1) * (Nat.choose (total_team_members - 2) regular_players)

theorem water_polo_team_selection :
  choose_team = 3268880 :=
sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l1197_119761


namespace NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l1197_119733

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_intersection_equals_singleton :
  (Set.compl M ∩ Set.compl N : Set (ℝ × ℝ)) = {(2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l1197_119733


namespace NUMINAMATH_CALUDE_intersection_triangle_is_right_angle_l1197_119767

/-- An ellipse with semi-major axis √m and semi-minor axis 1 -/
structure Ellipse (m : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / m + y^2 = 1)
  (m_gt_one : m > 1)

/-- A hyperbola with semi-major axis √n and semi-minor axis 1 -/
structure Hyperbola (n : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / n - y^2 = 1)
  (n_pos : n > 0)

/-- The foci of a conic section -/
structure Foci :=
  (F₁ F₂ : ℝ × ℝ)

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Theorem: The triangle formed by the foci and an intersection point of an ellipse and hyperbola with the same foci is a right triangle -/
theorem intersection_triangle_is_right_angle
  (m n : ℝ)
  (E : Ellipse m)
  (H : Hyperbola n)
  (F : Foci)
  (P : Point)
  (h₁ : E.x = P.1 ∧ E.y = P.2)  -- P is on the ellipse
  (h₂ : H.x = P.1 ∧ H.y = P.2)  -- P is on the hyperbola
  (h₃ : F.F₁ ≠ F.F₂)  -- The foci are distinct
  : ∃ (A B C : ℝ),
    (P.1 - F.F₁.1)^2 + (P.2 - F.F₁.2)^2 = A^2 ∧
    (P.1 - F.F₂.1)^2 + (P.2 - F.F₂.2)^2 = B^2 ∧
    (F.F₁.1 - F.F₂.1)^2 + (F.F₁.2 - F.F₂.2)^2 = C^2 ∧
    A^2 + B^2 = C^2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_triangle_is_right_angle_l1197_119767


namespace NUMINAMATH_CALUDE_central_cell_is_seven_l1197_119774

-- Define the grid
def Grid := Fin 3 → Fin 3 → Fin 9

-- Define adjacency
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ a.2.succ = b.2) ∨
  (a.1 = b.1 ∧ a.2 = b.2.succ) ∨
  (a.1.succ = b.1 ∧ a.2 = b.2) ∨
  (a.1 = b.1.succ ∧ a.2 = b.2)

-- Define consecutive numbers
def consecutive (a b : Fin 9) : Prop :=
  a.val.succ = b.val ∨ b.val.succ = a.val

-- Define the property of consecutive numbers being adjacent
def consecutiveAdjacent (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)

-- Define corner cells
def cornerCells : List (Fin 3 × Fin 3) :=
  [(0, 0), (0, 2), (2, 0), (2, 2)]

-- Define the sum of corner cells
def cornerSum (g : Grid) : Nat :=
  (cornerCells.map (fun (i, j) => (g i j).val)).sum

-- Define central cell
def centralCell : Fin 3 × Fin 3 := (1, 1)

-- Theorem statement
theorem central_cell_is_seven (g : Grid)
  (h1 : consecutiveAdjacent g)
  (h2 : cornerSum g = 18) :
  (g centralCell.1 centralCell.2).val = 7 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_is_seven_l1197_119774


namespace NUMINAMATH_CALUDE_calculation_proof_l1197_119709

theorem calculation_proof :
  (9.5 * 101 = 959.5) ∧
  (12.5 * 8.8 = 110) ∧
  (38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320) ∧
  (5.29 * 73 + 52.9 * 2.7 = 529) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1197_119709


namespace NUMINAMATH_CALUDE_remainder_theorem_f_of_one_eq_four_remainder_is_four_l1197_119794

-- Define the polynomial f(x) = x^15 + 3
def f (x : ℝ) : ℝ := x^15 + 3

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + f 1 := by
  sorry

-- Prove that f(1) = 4
theorem f_of_one_eq_four : f 1 = 4 := by
  sorry

-- Main theorem: The remainder when x^15 + 3 is divided by x-1 is 4
theorem remainder_is_four :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_f_of_one_eq_four_remainder_is_four_l1197_119794


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l1197_119739

/-- The cost of apples given a rate per half dozen -/
def appleCost (halfDozenRate : ℚ) (dozens : ℚ) : ℚ :=
  dozens * (2 * halfDozenRate)

theorem apple_cost_theorem (halfDozenRate : ℚ) :
  halfDozenRate = (4.80 : ℚ) →
  appleCost halfDozenRate 4 = (38.40 : ℚ) :=
by
  sorry

#eval appleCost (4.80 : ℚ) 4

end NUMINAMATH_CALUDE_apple_cost_theorem_l1197_119739


namespace NUMINAMATH_CALUDE_min_value_a_l1197_119707

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l1197_119707


namespace NUMINAMATH_CALUDE_consecutive_numbers_pairing_l1197_119773

theorem consecutive_numbers_pairing (n : ℕ) : 
  ∃ (p₁ p₂ : List (ℕ × ℕ)), 
    p₁ ≠ p₂ ∧ 
    (∀ i ∈ [0, 1, 2, 3, 4], 
      (n + 2*i) ∈ (p₁.map Prod.fst ++ p₁.map Prod.snd) ∧ 
      (n + 2*i + 1) ∈ (p₁.map Prod.fst ++ p₁.map Prod.snd)) ∧
    (∀ i ∈ [0, 1, 2, 3, 4], 
      (n + 2*i) ∈ (p₂.map Prod.fst ++ p₂.map Prod.snd) ∧ 
      (n + 2*i + 1) ∈ (p₂.map Prod.fst ++ p₂.map Prod.snd)) ∧
    p₁.length = 5 ∧ 
    p₂.length = 5 ∧ 
    (p₁.map (λ (a, b) => a * b)).sum = (p₂.map (λ (a, b) => a * b)).sum :=
by
  sorry


end NUMINAMATH_CALUDE_consecutive_numbers_pairing_l1197_119773


namespace NUMINAMATH_CALUDE_find_a_l1197_119750

theorem find_a : ∃ (a : ℝ), (∀ (x : ℝ), (2 * x - a ≤ -1) ↔ (x ≤ 1)) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_l1197_119750


namespace NUMINAMATH_CALUDE_f_properties_l1197_119732

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem f_properties (a : ℝ) :
  (∀ x > 1, Monotone (f (-2)))
  ∧ (∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then f a 1
       else if a > -2 * (exp 1)^2 then f a (sqrt (-a/2))
       else f a (exp 1))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1197_119732


namespace NUMINAMATH_CALUDE_line_equation_and_range_l1197_119749

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ := l.k * x + l.b

theorem line_equation_and_range (l : Line) 
  (h1 : l.y_at (-1) = 2)
  (h2 : l.y_at 2 = 5) :
  (∀ x, l.y_at x = x + 3) ∧ 
  (∀ x, l.y_at x > 0 ↔ x > -3) := by
  sorry


end NUMINAMATH_CALUDE_line_equation_and_range_l1197_119749


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l1197_119713

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 10) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 10 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l1197_119713


namespace NUMINAMATH_CALUDE_youtube_video_length_l1197_119701

/-- Represents the duration of a YouTube video session in seconds -/
def YouTubeSession (ad1 ad2 video1 video2 pause totalTime : ℕ) (lastTwoEqual : Bool) : Prop :=
  let firstVideoTotal := ad1 + 120  -- 2 minutes = 120 seconds
  let secondVideoTotal := ad2 + 270  -- 4 minutes 30 seconds = 270 seconds
  let remainingTime := totalTime - (firstVideoTotal + secondVideoTotal)
  let lastTwoVideosTime := remainingTime - pause
  lastTwoEqual ∧ 
  (lastTwoVideosTime / 2 = 495) ∧
  (totalTime = 1500)

theorem youtube_video_length 
  (ad1 ad2 video1 video2 pause totalTime : ℕ) 
  (lastTwoEqual : Bool) 
  (h : YouTubeSession ad1 ad2 video1 video2 pause totalTime lastTwoEqual) :
  ∃ (lastVideoLength : ℕ), lastVideoLength = 495 :=
sorry

end NUMINAMATH_CALUDE_youtube_video_length_l1197_119701


namespace NUMINAMATH_CALUDE_inequality_proof_l1197_119751

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1197_119751


namespace NUMINAMATH_CALUDE_cost_of_candies_l1197_119706

/-- The cost of buying lollipops and chocolates -/
theorem cost_of_candies (lollipop_cost : ℕ) (chocolate_cost : ℕ) 
  (lollipop_count : ℕ) (chocolate_count : ℕ) : 
  lollipop_cost = 3 →
  chocolate_cost = 2 →
  lollipop_count = 500 →
  chocolate_count = 300 →
  (lollipop_cost * lollipop_count + chocolate_cost * chocolate_count : ℕ) / 100 = 21 :=
by
  sorry

#check cost_of_candies

end NUMINAMATH_CALUDE_cost_of_candies_l1197_119706


namespace NUMINAMATH_CALUDE_tv_price_increase_l1197_119757

theorem tv_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.2 = 1 + 56.00000000000001 / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l1197_119757


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverses_problem_l1197_119758

theorem additive_multiplicative_inverses_problem (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are additive inverses
  (h2 : c * d = 1)  -- c and d are multiplicative inverses
  (h3 : abs m = 1)  -- absolute value of m is 1
  : (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 := by
  sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverses_problem_l1197_119758


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1197_119768

/-- The number of ways to arrange 3 people on 6 chairs in a row, 
    such that no two people sit next to each other -/
def arrangement_count : ℕ := 24

/-- The number of chairs in the row -/
def total_chairs : ℕ := 6

/-- The number of people to be seated -/
def people_count : ℕ := 3

/-- Theorem stating that the number of arrangements 
    satisfying the given conditions is 24 -/
theorem arrangement_theorem : 
  arrangement_count = 
    (Nat.factorial people_count) * (total_chairs - people_count - (people_count - 1)) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1197_119768


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_a_eq_neg_twelve_l1197_119790

theorem infinite_solutions_iff_a_eq_neg_twelve (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_a_eq_neg_twelve_l1197_119790


namespace NUMINAMATH_CALUDE_customers_who_left_l1197_119723

theorem customers_who_left (initial_customers : ℕ) (new_customers : ℕ) (final_customers : ℕ) :
  initial_customers = 19 →
  new_customers = 36 →
  final_customers = 41 →
  initial_customers - (initial_customers - new_customers - final_customers) + new_customers = final_customers :=
by
  sorry

end NUMINAMATH_CALUDE_customers_who_left_l1197_119723


namespace NUMINAMATH_CALUDE_problem_solution_l1197_119748

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 - f a 2 = 1) :
  (∃ m_lower m_upper : ℝ, m_lower = 2/3 ∧ m_upper = 7 ∧
    ∀ m : ℝ, m_lower < m ∧ m < m_upper ↔ f a (3*m - 2) < f a (2*m + 5)) ∧
  (∃ x : ℝ, x = 4 ∧ f a (x - 2/x) = Real.log (7/2) / Real.log (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1197_119748


namespace NUMINAMATH_CALUDE_weight_of_barium_fluoride_l1197_119715

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of Barium atoms in BaF2 -/
def num_Ba : ℕ := 1

/-- The number of Fluorine atoms in BaF2 -/
def num_F : ℕ := 2

/-- The number of moles of BaF2 -/
def num_moles : ℝ := 3

/-- Theorem: The weight of 3 moles of Barium fluoride (BaF2) is 525.99 grams -/
theorem weight_of_barium_fluoride :
  (num_moles * (num_Ba * atomic_weight_Ba + num_F * atomic_weight_F)) = 525.99 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_barium_fluoride_l1197_119715


namespace NUMINAMATH_CALUDE_bob_net_increase_theorem_l1197_119731

/-- Calculates the net increase in weekly earnings given a raise, work hours, and benefit reduction --/
def netIncreaseInWeeklyEarnings (hourlyRaise : ℚ) (weeklyHours : ℕ) (monthlyBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := hourlyRaise * weeklyHours
  let weeklyBenefitReduction := monthlyBenefitReduction / 4
  weeklyRaise - weeklyBenefitReduction

/-- Theorem stating that given the specified conditions, the net increase in weekly earnings is $5 --/
theorem bob_net_increase_theorem :
  netIncreaseInWeeklyEarnings (1/2) 40 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bob_net_increase_theorem_l1197_119731


namespace NUMINAMATH_CALUDE_order_amount_for_88200_l1197_119736

/-- Calculates the discount rate based on the order quantity -/
def discount_rate (x : ℕ) : ℚ :=
  if x < 250 then 0
  else if x < 500 then 1/20
  else if x < 1000 then 1/10
  else 3/20

/-- Calculates the payable amount given the order quantity and unit price -/
def payable_amount (x : ℕ) (A : ℚ) : ℚ :=
  A * x * (1 - discount_rate x)

/-- The unit price determined from the given condition -/
def unit_price : ℚ := 100

theorem order_amount_for_88200 :
  payable_amount 980 unit_price = 88200 :=
sorry

end NUMINAMATH_CALUDE_order_amount_for_88200_l1197_119736


namespace NUMINAMATH_CALUDE_keith_cantaloupes_l1197_119747

/-- The number of cantaloupes grown by Keith, given the total number of cantaloupes
    and the numbers grown by Fred and Jason. -/
theorem keith_cantaloupes (total : ℕ) (fred : ℕ) (jason : ℕ) 
    (h_total : total = 65) 
    (h_fred : fred = 16) 
    (h_jason : jason = 20) : 
  total - (fred + jason) = 29 := by
  sorry

end NUMINAMATH_CALUDE_keith_cantaloupes_l1197_119747


namespace NUMINAMATH_CALUDE_product_is_2008th_power_l1197_119740

theorem product_is_2008th_power : ∃ (a b c : ℕ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  ((a = (b + c) / 2) ∨ (b = (a + c) / 2) ∨ (c = (a + b) / 2)) ∧
  (∃ (n : ℕ), a * b * c = n ^ 2008) :=
by sorry

end NUMINAMATH_CALUDE_product_is_2008th_power_l1197_119740


namespace NUMINAMATH_CALUDE_soccer_campers_l1197_119759

theorem soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : football = 32) : 
  total - (basketball + football) = 32 := by
  sorry

end NUMINAMATH_CALUDE_soccer_campers_l1197_119759


namespace NUMINAMATH_CALUDE_binomial_150_150_l1197_119727

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1197_119727


namespace NUMINAMATH_CALUDE_simplify_rational_function_l1197_119776

theorem simplify_rational_function (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_rational_function_l1197_119776


namespace NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l1197_119744

/-- A function that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- The theorem statement -/
theorem largest_four_digit_number_with_conditions :
  ∃ (n : ℕ),
    n = 8910 ∧
    1000 ≤ n ∧ n < 10000 ∧
    allDigitsDifferent n ∧
    n % 2 = 0 ∧ n % 5 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
    ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ allDigitsDifferent m ∧
      m % 2 = 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l1197_119744


namespace NUMINAMATH_CALUDE_number_of_elements_l1197_119703

theorem number_of_elements (incorrect_avg : ℝ) (correct_avg : ℝ) (difference : ℝ) : 
  incorrect_avg = 16 → correct_avg = 17 → difference = 10 →
  ∃ n : ℕ, n * correct_avg = n * incorrect_avg + difference ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_elements_l1197_119703


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l1197_119787

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![1, 4], ![-2, -7]] →
  (A^3)⁻¹ = ![![41, 140], ![-90, -335]] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l1197_119787


namespace NUMINAMATH_CALUDE_some_students_not_honor_society_l1197_119720

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Scholarship : U → Prop)
variable (HonorSociety : U → Prop)

-- State the theorem
theorem some_students_not_honor_society :
  (∃ x, Student x ∧ ¬Scholarship x) →
  (∀ x, HonorSociety x → Scholarship x) →
  (∃ x, Student x ∧ ¬HonorSociety x) :=
by
  sorry


end NUMINAMATH_CALUDE_some_students_not_honor_society_l1197_119720


namespace NUMINAMATH_CALUDE_min_bricks_for_cube_l1197_119704

/-- The width of a brick in centimeters -/
def brick_width : ℕ := 18

/-- The depth of a brick in centimeters -/
def brick_depth : ℕ := 12

/-- The height of a brick in centimeters -/
def brick_height : ℕ := 9

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℕ := brick_width * brick_depth * brick_height

/-- The side length of the smallest cube that can be formed using the bricks -/
def cube_side_length : ℕ := Nat.lcm (Nat.lcm brick_width brick_depth) brick_height

/-- The volume of the smallest cube that can be formed using the bricks -/
def cube_volume : ℕ := cube_side_length ^ 3

/-- The theorem stating the minimum number of bricks required to make a cube -/
theorem min_bricks_for_cube : cube_volume / brick_volume = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_bricks_for_cube_l1197_119704


namespace NUMINAMATH_CALUDE_min_value_theorem_l1197_119728

/-- The minimum value of 1/m + 2/n given the constraints -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1197_119728


namespace NUMINAMATH_CALUDE_min_xy_value_l1197_119714

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) :
  x * y ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_xy_value_l1197_119714


namespace NUMINAMATH_CALUDE_car_travel_time_l1197_119764

theorem car_travel_time (distance : ℝ) (speed : ℝ) (time_ratio : ℝ) (initial_time : ℝ) : 
  distance = 324 →
  speed = 36 →
  time_ratio = 3 / 2 →
  distance = speed * (time_ratio * initial_time) →
  initial_time = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l1197_119764


namespace NUMINAMATH_CALUDE_problem_solution_l1197_119752

theorem problem_solution : ∃! x : ℝ, 0.8 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1197_119752


namespace NUMINAMATH_CALUDE_two_pump_fill_time_l1197_119779

/-- The time taken for two pumps to fill a tank together -/
theorem two_pump_fill_time (small_pump_rate large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 2)
  (h2 : large_pump_rate = 3)
  (h3 : small_pump_rate > 0)
  (h4 : large_pump_rate > 0) :
  1 / (small_pump_rate + large_pump_rate) = 1 / 3.5 :=
by sorry

end NUMINAMATH_CALUDE_two_pump_fill_time_l1197_119779


namespace NUMINAMATH_CALUDE_total_spent_is_36_98_l1197_119737

/-- Calculates the total amount spent on video games --/
def total_spent (football_price : ℝ) (football_discount : ℝ) 
                (strategy_price : ℝ) (strategy_tax : ℝ)
                (batman_price_euro : ℝ) (exchange_rate : ℝ) : ℝ :=
  let football_discounted := football_price * (1 - football_discount)
  let strategy_with_tax := strategy_price * (1 + strategy_tax)
  let batman_price_usd := batman_price_euro * exchange_rate
  football_discounted + strategy_with_tax + batman_price_usd

/-- Theorem stating the total amount spent on video games --/
theorem total_spent_is_36_98 :
  total_spent 16 0.1 9.46 0.05 11 1.15 = 36.98 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_36_98_l1197_119737


namespace NUMINAMATH_CALUDE_total_bread_making_time_l1197_119712

/-- The time it takes to make bread, given the time for rising, kneading, and baking. -/
def bread_making_time (rise_time : ℕ) (kneading_time : ℕ) (baking_time : ℕ) : ℕ :=
  2 * rise_time + kneading_time + baking_time

/-- Theorem stating that the total time to make bread is 280 minutes. -/
theorem total_bread_making_time :
  bread_making_time 120 10 30 = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_making_time_l1197_119712


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l1197_119788

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0, 1, 2, or 3 members -/
def small_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2) + (n.choose 3)

/-- The maximum number of points that can lie on a semicircle -/
def max_semicircle : ℕ := n / 2 + 1

/-- The number of subsets that lie on a semicircle -/
def semicircle_subsets : ℕ := 2^max_semicircle - 1

/-- Conservative estimate of subsets to exclude due to lying on the same semicircle -/
def conservative_exclusion : ℕ := 500

/-- The number of distinct convex polygons with 4 or more sides -/
def distinct_polygons : ℕ := total_subsets - small_subsets - semicircle_subsets - conservative_exclusion

theorem distinct_polygons_count :
  distinct_polygons = 31437 :=
sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l1197_119788


namespace NUMINAMATH_CALUDE_prob_rain_smallest_n_l1197_119724

/-- The probability of rain on day n, given it doesn't rain on day 0 -/
def prob_rain (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | k + 1 => 0.5 * prob_rain k + 0.25

/-- The smallest positive integer n such that the probability of rain n days from today is greater than 49.9% -/
def smallest_n : ℕ := 9

theorem prob_rain_smallest_n :
  (∀ k < smallest_n, prob_rain k ≤ 0.499) ∧
  prob_rain smallest_n > 0.499 :=
by sorry

#eval smallest_n

end NUMINAMATH_CALUDE_prob_rain_smallest_n_l1197_119724


namespace NUMINAMATH_CALUDE_component_usage_impossibility_l1197_119721

theorem component_usage_impossibility (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧
    (2 * x + y = 2 * p + q + 1) ∧
    (y + z = q + r) := by
  sorry

end NUMINAMATH_CALUDE_component_usage_impossibility_l1197_119721


namespace NUMINAMATH_CALUDE_apple_orange_difference_l1197_119700

theorem apple_orange_difference (total : Nat) (apples : Nat) (h1 : total = 301) (h2 : apples = 164) (h3 : apples > total - apples) : apples - (total - apples) = 27 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_difference_l1197_119700


namespace NUMINAMATH_CALUDE_number_divided_and_subtracted_l1197_119762

theorem number_divided_and_subtracted (x : ℝ) : x = 4.5 → x / 3 = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_and_subtracted_l1197_119762


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_negative_one_l1197_119777

theorem sum_of_powers_equals_negative_one :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_negative_one_l1197_119777


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l1197_119718

/-- Calculate the total cost for Jasmine's purchase of coffee beans and milk --/
theorem jasmine_purchase_cost :
  let coffee_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let discount_rate : ℚ := 1/10
  let tax_rate : ℚ := 2/25

  let total_before_discount : ℚ := coffee_pounds * coffee_price_per_pound + milk_gallons * milk_price_per_gallon
  let discount : ℚ := discount_rate * total_before_discount
  let discounted_price : ℚ := total_before_discount - discount
  let taxes : ℚ := tax_rate * discounted_price
  let final_amount : ℚ := discounted_price + taxes

  final_amount = 1652/100 := by sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l1197_119718


namespace NUMINAMATH_CALUDE_floor_plus_x_equation_l1197_119784

theorem floor_plus_x_equation (x : ℝ) : (⌊x⌋ : ℝ) + x = 20.5 ↔ x = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_equation_l1197_119784


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1197_119743

theorem similar_triangles_height (h1 h2 : ℝ) (A1 A2 : ℝ) : 
  h1 > 0 → h2 > 0 → A1 > 0 → A2 > 0 →
  (A2 / A1 = 9) →  -- Area ratio
  h1 = 5 →         -- Height of smaller triangle
  (h2 / h1)^2 = (A2 / A1) →  -- Similarity condition
  h2 = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1197_119743


namespace NUMINAMATH_CALUDE_equation_equality_l1197_119719

theorem equation_equality (x : ℝ) : 
  4 * x^4 + x^3 - 2*x + 5 + (-4 * x^4 + x^3 - 7 * x^2 + 2*x - 1) = 2 * x^3 - 7 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1197_119719


namespace NUMINAMATH_CALUDE_total_participants_is_280_l1197_119760

/-- The number of students who participated in at least one competition -/
def total_participants (math physics chem math_physics math_chem phys_chem all_three : ℕ) : ℕ :=
  math + physics + chem - math_physics - math_chem - phys_chem + all_three

/-- Theorem stating that the total number of participants is 280 given the conditions -/
theorem total_participants_is_280 :
  total_participants 203 179 165 143 116 97 89 = 280 := by
  sorry

#eval total_participants 203 179 165 143 116 97 89

end NUMINAMATH_CALUDE_total_participants_is_280_l1197_119760


namespace NUMINAMATH_CALUDE_probability_even_distinct_digits_l1197_119702

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

def count_valid_numbers : ℕ := sorry

theorem probability_even_distinct_digits :
  (count_valid_numbers : ℚ) / (9999 - 1000 + 1 : ℚ) = 343 / 1125 := by sorry

end NUMINAMATH_CALUDE_probability_even_distinct_digits_l1197_119702


namespace NUMINAMATH_CALUDE_whiskers_cat_school_total_l1197_119708

/-- Represents the number of cats that can perform a specific trick or combination of tricks -/
structure CatTricks where
  jump : ℕ
  sit : ℕ
  playDead : ℕ
  fetch : ℕ
  jumpSit : ℕ
  sitPlayDead : ℕ
  playDeadFetch : ℕ
  fetchJump : ℕ
  jumpSitPlayDead : ℕ
  sitPlayDeadFetch : ℕ
  playDeadFetchJump : ℕ
  jumpFetchSit : ℕ
  allFour : ℕ
  none : ℕ

/-- Calculates the total number of cats in the Whisker's Cat School -/
def totalCats (tricks : CatTricks) : ℕ :=
  let exclusiveJump := tricks.jump - (tricks.jumpSit + tricks.fetchJump + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveSit := tricks.sit - (tricks.jumpSit + tricks.sitPlayDead + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusivePlayDead := tricks.playDead - (tricks.sitPlayDead + tricks.playDeadFetch + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveFetch := tricks.fetch - (tricks.playDeadFetch + tricks.fetchJump + tricks.sitPlayDeadFetch + tricks.allFour)
  exclusiveJump + exclusiveSit + exclusivePlayDead + exclusiveFetch +
  tricks.jumpSit + tricks.sitPlayDead + tricks.playDeadFetch + tricks.fetchJump +
  tricks.jumpSitPlayDead + tricks.sitPlayDeadFetch + tricks.playDeadFetchJump + tricks.jumpFetchSit +
  tricks.allFour + tricks.none

/-- The specific number of cats for each trick or combination at the Whisker's Cat School -/
def whiskersCatSchool : CatTricks :=
  { jump := 60
  , sit := 40
  , playDead := 35
  , fetch := 45
  , jumpSit := 20
  , sitPlayDead := 15
  , playDeadFetch := 10
  , fetchJump := 18
  , jumpSitPlayDead := 5
  , sitPlayDeadFetch := 3
  , playDeadFetchJump := 7
  , jumpFetchSit := 10
  , allFour := 2
  , none := 12 }

/-- Theorem stating that the total number of cats at the Whisker's Cat School is 143 -/
theorem whiskers_cat_school_total : totalCats whiskersCatSchool = 143 := by
  sorry

end NUMINAMATH_CALUDE_whiskers_cat_school_total_l1197_119708


namespace NUMINAMATH_CALUDE_least_integer_with_twelve_factors_l1197_119770

theorem least_integer_with_twelve_factors : 
  ∀ n : ℕ, n > 0 → (∃ f : Finset ℕ, f = {d : ℕ | d ∣ n ∧ d > 0} ∧ f.card = 12) → n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_twelve_factors_l1197_119770


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1197_119742

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x^2 < 2*x + 3 → x ≥ 0 ∧ 0^2 < 2*0 + 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1197_119742


namespace NUMINAMATH_CALUDE_regular_pyramid_from_equal_edges_l1197_119766

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a common point (apex) --/
structure Pyramid where
  base : Set Point
  apex : Point
  lateral_faces : Set (Set Point)

/-- A regular pyramid is a pyramid with a regular polygon base and congruent triangular faces --/
def IsRegularPyramid (p : Pyramid) : Prop := sorry

/-- All edges of a pyramid have equal length --/
def AllEdgesEqual (p : Pyramid) : Prop := sorry

/-- Theorem: If all edges of a pyramid are equal, then it is a regular pyramid --/
theorem regular_pyramid_from_equal_edges (p : Pyramid) :
  AllEdgesEqual p → IsRegularPyramid p := by sorry

end NUMINAMATH_CALUDE_regular_pyramid_from_equal_edges_l1197_119766


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1197_119780

theorem beef_weight_before_processing 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : loss_percentage = 40) 
  (h2 : final_weight = 240) 
  (h3 : final_weight = initial_weight * (1 - loss_percentage / 100)) : 
  initial_weight = 400 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1197_119780


namespace NUMINAMATH_CALUDE_fifty_eight_impossible_l1197_119783

/-- Represents the population of Rivertown -/
structure RivertownPopulation where
  people : ℕ
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  chickens : ℕ
  people_dog_ratio : people = 5 * dogs
  cat_rabbit_ratio : cats = 2 * rabbits
  chicken_people_ratio : chickens = 4 * people

/-- The total population of Rivertown -/
def totalPopulation (pop : RivertownPopulation) : ℕ :=
  pop.people + pop.dogs + pop.cats + pop.rabbits + pop.chickens

/-- Theorem stating that 58 cannot be the total population of Rivertown -/
theorem fifty_eight_impossible (pop : RivertownPopulation) : totalPopulation pop ≠ 58 := by
  sorry

end NUMINAMATH_CALUDE_fifty_eight_impossible_l1197_119783


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1197_119782

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1197_119782


namespace NUMINAMATH_CALUDE_solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l1197_119789

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f_leq_4 :
  {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the range of m where f(x) > m^2 + m always holds
theorem range_of_m_f_gt_m_squared_plus_m :
  {m : ℝ | ∀ x, f x > m^2 + m} = {m : ℝ | -2 < m ∧ m < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l1197_119789


namespace NUMINAMATH_CALUDE_fraction_comparison_l1197_119786

theorem fraction_comparison (m : ℕ) (h : m = 23^1973) :
  (23^1873 + 1) / (23^1974 + 1) > (23^1974 + 1) / (23^1975 + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1197_119786


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1197_119772

/-- The radius of a sphere tangent to the bases and lateral surface of a truncated cone --/
theorem sphere_radius_in_truncated_cone (R r : ℝ) (hR : R = 24) (hr : r = 5) :
  ∃ (radius : ℝ), radius > 0 ∧ radius^2 = (R - r)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1197_119772


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1197_119797

/-- The line l intersects with the circle C if the distance from the center of C to l is less than the radius of C. -/
theorem line_intersects_circle (m : ℝ) : 
  let l : Set (ℝ × ℝ) := {(x, y) | m * x - y + 1 = 0}
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y-1)^2 = 5}
  let center : ℝ × ℝ := (0, 1)
  let radius : ℝ := Real.sqrt 5
  let distance_to_line (p : ℝ × ℝ) : ℝ := 
    abs (m * p.1 - p.2 + 1) / Real.sqrt (m^2 + 1)
  distance_to_line center < radius → 
  ∃ p, p ∈ l ∧ p ∈ C := by
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1197_119797


namespace NUMINAMATH_CALUDE_farm_chicken_ratio_l1197_119716

/-- Given a farm with chickens, prove the ratio of hens to roosters -/
theorem farm_chicken_ratio (total : ℕ) (hens : ℕ) (X : ℕ) : 
  total = 75 → 
  hens = 67 → 
  hens = X * (total - hens) - 5 → 
  X = 9 := by
sorry

end NUMINAMATH_CALUDE_farm_chicken_ratio_l1197_119716


namespace NUMINAMATH_CALUDE_farm_trip_chaperones_l1197_119710

theorem farm_trip_chaperones (num_students : ℕ) (student_fee adult_fee total_fee : ℚ) : 
  num_students = 35 →
  student_fee = 5 →
  adult_fee = 6 →
  total_fee = 199 →
  ∃ (num_adults : ℕ), num_adults * adult_fee + num_students * student_fee = total_fee ∧ num_adults = 4 :=
by sorry

end NUMINAMATH_CALUDE_farm_trip_chaperones_l1197_119710


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1197_119734

theorem tan_pi_minus_alpha (α : Real) 
  (h1 : α > π / 2) 
  (h2 : α < π) 
  (h3 : 3 * Real.cos (2 * α) - Real.sin α = 2) : 
  Real.tan (π - α) = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1197_119734


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l1197_119746

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed Christina_speed lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : jack_speed = 7)
  (h2 : Christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) :
  (jack_speed + Christina_speed) * (lindy_distance / lindy_speed) = 150 := by
  sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l1197_119746


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1197_119753

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1197_119753


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1197_119729

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 10th term of a geometric sequence with first term 2 and second term 5/2 -/
theorem tenth_term_of_geometric_sequence :
  let a : ℚ := 2
  let second_term : ℚ := 5/2
  let r : ℚ := second_term / a
  geometric_term a r 10 = 3906250/262144 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1197_119729


namespace NUMINAMATH_CALUDE_total_trophies_after_seven_years_l1197_119778

def michael_initial_trophies : ℕ := 100
def michael_yearly_increase : ℕ := 200
def years : ℕ := 7
def jack_multiplier : ℕ := 20

def michael_final_trophies : ℕ := michael_initial_trophies + michael_yearly_increase * years
def jack_final_trophies : ℕ := jack_multiplier * michael_initial_trophies + michael_final_trophies

theorem total_trophies_after_seven_years :
  michael_final_trophies + jack_final_trophies = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_after_seven_years_l1197_119778


namespace NUMINAMATH_CALUDE_bob_pays_48_percent_l1197_119796

-- Define the suggested retail price
variable (P : ℝ)

-- Define the marked price as 80% of the suggested retail price
def markedPrice (P : ℝ) : ℝ := 0.8 * P

-- Define Bob's purchase price as 60% of the marked price
def bobPrice (P : ℝ) : ℝ := 0.6 * markedPrice P

-- Theorem statement
theorem bob_pays_48_percent (P : ℝ) (h : P > 0) : 
  bobPrice P / P = 0.48 := by
sorry

end NUMINAMATH_CALUDE_bob_pays_48_percent_l1197_119796


namespace NUMINAMATH_CALUDE_max_rectangles_in_6x6_square_l1197_119717

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the maximum number of rectangles that can fit in a square -/
def max_rectangles_in_square (r : Rectangle) (s : Square) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 4×1 rectangles in a 6×6 square -/
theorem max_rectangles_in_6x6_square :
  let r : Rectangle := ⟨4, 1⟩
  let s : Square := ⟨6⟩
  max_rectangles_in_square r s = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_6x6_square_l1197_119717


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1197_119725

theorem min_value_of_expression (x : ℝ) : (3*x^2 + 6*x + 5) / ((1/2)*x^2 + x + 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1197_119725


namespace NUMINAMATH_CALUDE_at_least_one_female_selection_l1197_119765

-- Define the total number of athletes
def total_athletes : ℕ := 10

-- Define the number of male athletes
def male_athletes : ℕ := 6

-- Define the number of female athletes
def female_athletes : ℕ := 4

-- Define the number of athletes to be selected
def selected_athletes : ℕ := 5

-- Theorem statement
theorem at_least_one_female_selection :
  (Nat.choose total_athletes selected_athletes) - (Nat.choose male_athletes selected_athletes) = 246 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_female_selection_l1197_119765


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1197_119726

-- Define set A
def A : Set ℝ := {y | ∃ x, y = -x^2 + 2*x - 1}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x + 1)}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-1/2) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1197_119726


namespace NUMINAMATH_CALUDE_garrett_granola_bars_l1197_119771

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_bars : ℕ := 8

/-- The total number of granola bars Garrett bought -/
def total_bars : ℕ := oatmeal_raisin_bars + peanut_bars

theorem garrett_granola_bars : total_bars = 14 := by sorry

end NUMINAMATH_CALUDE_garrett_granola_bars_l1197_119771


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1197_119795

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes x ± 2y = 0 is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1197_119795


namespace NUMINAMATH_CALUDE_orange_crates_count_l1197_119793

theorem orange_crates_count :
  ∀ (num_crates : ℕ),
    (∀ (crate : ℕ), crate ≤ num_crates → 150 * num_crates + 16 * 30 = 2280) →
    num_crates = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_crates_count_l1197_119793


namespace NUMINAMATH_CALUDE_jills_nails_count_l1197_119785

theorem jills_nails_count : ∃ N : ℕ,
  N > 0 ∧
  (8 : ℝ) / N * 100 - ((N : ℝ) - 14) / N * 100 = 10 ∧
  6 + 8 + (N - 14) = N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jills_nails_count_l1197_119785
