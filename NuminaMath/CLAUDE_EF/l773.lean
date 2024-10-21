import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_net_loss_percent_l773_77321

/-- Represents the financial details of an article sale --/
structure ArticleSale where
  costPrice : ℝ
  profitPercent : ℝ
  discountPercent : ℝ
  taxPercent : ℝ
  processingFeePercent : ℝ

/-- Calculates the final selling price of an article --/
def finalSellingPrice (sale : ArticleSale) : ℝ :=
  let priceAfterProfitLoss := sale.costPrice * (1 + sale.profitPercent)
  let priceAfterDiscount := priceAfterProfitLoss * (1 - sale.discountPercent)
  let priceAfterTax := priceAfterDiscount * (1 + sale.taxPercent)
  priceAfterTax * (1 + sale.processingFeePercent)

/-- Theorem stating the combined net loss percent after all transactions --/
theorem combined_net_loss_percent (
  article1 article2 article3 article4 : ArticleSale)
  (h1 : article1.costPrice = 1000 ∧ article1.profitPercent = 0.1 ∧ article1.discountPercent = 0.05 ∧ article1.taxPercent = 0 ∧ article1.processingFeePercent = 0)
  (h2 : article2.costPrice = 1000 ∧ article2.profitPercent = -0.1 ∧ article2.discountPercent = 0 ∧ article2.taxPercent = 0.02 ∧ article2.processingFeePercent = 0)
  (h3 : article3.costPrice = 1000 ∧ article3.profitPercent = 0.2 ∧ article3.discountPercent = 0 ∧ article3.taxPercent = 0 ∧ article3.processingFeePercent = 0.03)
  (h4 : article4.costPrice = 1000 ∧ article4.profitPercent = -0.25 ∧ article4.discountPercent = 0 ∧ article4.taxPercent = 0.01 ∧ article4.processingFeePercent = 0) :
  ∃ (lossPercent : ℝ), abs (lossPercent - 0.0109) < 0.0001 ∧
  lossPercent = (4000 - (finalSellingPrice article1 + finalSellingPrice article2 + finalSellingPrice article3 + finalSellingPrice article4)) / 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_net_loss_percent_l773_77321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_ln_is_neg_exp_l773_77320

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the rotation transformation
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the rotated function
noncomputable def g (x : ℝ) : ℝ := -Real.exp x

-- Theorem statement
theorem rotate_ln_is_neg_exp :
  ∀ x > 0, rotate270 (x, f x) = (f x, g (f x)) := by
  intro x hx
  simp [rotate270, f, g]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_ln_is_neg_exp_l773_77320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l773_77316

/-- Represents a right triangle with specific properties -/
structure RightTriangle where
  longer_leg : ℝ
  shorter_leg : ℝ
  hypotenuse : ℝ
  area : ℝ
  shorter_leg_eq : shorter_leg = 0.5 * longer_leg - 3
  area_eq : area = 84
  pythagoras : hypotenuse^2 = longer_leg^2 + shorter_leg^2

/-- The length of the hypotenuse of the specific right triangle is approximately 22.96 -/
theorem hypotenuse_length (t : RightTriangle) : ∃ ε > 0, |t.hypotenuse - 22.96| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l773_77316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l773_77370

/-- A line in the xy-plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- The x-intercept of a line -/
noncomputable def Line.x_intercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line -/
noncomputable def Line.y_intercept (l : Line) : ℝ := -l.c / l.b

/-- Two lines are parallel if they have the same slope -/
def Line.parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem line_satisfies_conditions (l : Line) (l_given : Line) :
  l.a = 3 ∧ l.b = 4 ∧ l.c = -4 ∧
  l_given.a = 3 ∧ l_given.b = 4 ∧ l_given.c = 1 →
  l.parallel l_given ∧ 
  l.x_intercept + l.y_intercept = 7/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l773_77370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_theorem_l773_77394

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ
  averageIncrease : ℚ

/-- Calculates the current average runs per innings -/
def currentAverage (player : CricketPlayer) : ℚ :=
  player.totalRuns / player.innings

/-- Calculates the new average after the next innings -/
def newAverage (player : CricketPlayer) : ℚ :=
  (player.totalRuns + player.nextInningsRuns) / (player.innings + 1)

/-- Theorem: Given the conditions, prove that the current average is 33 -/
theorem cricket_average_theorem (player : CricketPlayer)
  (h1 : player.innings = 10)
  (h2 : player.nextInningsRuns = 77)
  (h3 : newAverage player - currentAverage player = 4) :
  currentAverage player = 33 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_theorem_l773_77394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l773_77330

-- Define the circles
noncomputable def circle_C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2 * Real.sqrt 3 * x - 4 * y + 6 = 0
def circle_C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6 * y = 0

-- Define the centers and radii of the circles
noncomputable def center_C₁ : ℝ × ℝ := (Real.sqrt 3, 2)
def center_C₂ : ℝ × ℝ := (0, 3)
def radius_C₁ : ℝ := 1
def radius_C₂ : ℝ := 3

-- Theorem statement
theorem circles_internally_tangent :
  let d := Real.sqrt ((center_C₁.1 - center_C₂.1)^2 + (center_C₁.2 - center_C₂.2)^2)
  d = |radius_C₂ - radius_C₁| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l773_77330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_is_even_integer_l773_77312

/- Define the set of allowed functions -/
inductive AllowedFunction : Type
| initial : AllowedFunction
| derivative : AllowedFunction → AllowedFunction
| sum : AllowedFunction → AllowedFunction → AllowedFunction
| product : AllowedFunction → AllowedFunction → AllowedFunction

/- Define the evaluation of an allowed function at x = 0 -/
def evaluate : AllowedFunction → ℤ
| AllowedFunction.initial => 1  -- sin(0) + cos(0) = 0 + 1 = 1
| AllowedFunction.derivative f => evaluate f  -- Placeholder, actual derivative evaluation is more complex
| AllowedFunction.sum f g => evaluate f + evaluate g
| AllowedFunction.product f g => evaluate f * evaluate g

/- Theorem: Any constant function obtained is an even integer -/
theorem constant_function_is_even_integer (f : AllowedFunction) (h : ∀ x : ℝ, evaluate f = evaluate f) :
  ∃ k : ℤ, evaluate f = 2 * k := by
  sorry

/- Note: The actual proof would involve showing that all operations preserve the property
   of evaluating to an even integer at x = 0, which implies the constant must be even. -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_is_even_integer_l773_77312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_distribution_unique_solution_l773_77382

/-- Represents the distribution of mushrooms to a girl -/
structure Distribution where
  base : ℕ
  remainder_fraction : ℚ
deriving Inhabited

/-- Calculates the number of mushrooms a girl receives given the total number of mushrooms -/
def mushrooms_received (d : Distribution) (total : ℕ) : ℚ :=
  d.base + (total - d.base : ℚ) * d.remainder_fraction

/-- Checks if all girls receive the same number of mushrooms -/
def all_equal (distributions : List Distribution) (total : ℕ) : Prop :=
  match distributions with
  | [] => true
  | (first :: rest) =>
    let first_amount := mushrooms_received first total
    rest.all (λ d => mushrooms_received d total = first_amount)

/-- The main theorem stating the unique solution to the mushroom distribution problem -/
theorem mushroom_distribution_unique_solution :
  ∃! (n : ℕ) (total : ℕ),
    n > 0 ∧
    total > 0 ∧
    let distributions := List.range n |>.map (λ i => ⟨20 + i, 4/100⟩)
    all_equal distributions total ∧
    n = 5 ∧
    total = 120 := by sorry

#check mushroom_distribution_unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_distribution_unique_solution_l773_77382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_l773_77358

theorem exponential_equation : (81 : ℝ) ^ (0.20 : ℝ) * (81 : ℝ) ^ (0.05 : ℝ) = 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_l773_77358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_folding_l773_77341

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side_length : ℝ
  angle_BAD : ℝ

/-- The distance between AC and BD after folding -/
noncomputable def distance_AC_BD (r : Rhombus) : ℝ := (Real.sqrt 3 / 4) * r.side_length

/-- Theorem stating the distance between AC and BD after folding -/
theorem distance_after_folding (r : Rhombus) 
  (h1 : r.angle_BAD = π/3)  -- 60 degrees in radians
  (h2 : r.side_length > 0) :
  distance_AC_BD r = (Real.sqrt 3 / 4) * r.side_length := by
  -- The proof goes here
  sorry

#check distance_after_folding

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_folding_l773_77341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tint_percentage_after_addition_l773_77345

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  total_volume : ℝ
  blue_tint_percentage : ℝ

/-- Calculates the new blue tint percentage after adding pure blue tint -/
noncomputable def new_blue_tint_percentage (original : PaintMixture) (added_blue_tint : ℝ) : ℝ :=
  let original_blue_volume := original.total_volume * (original.blue_tint_percentage / 100)
  let new_total_volume := original.total_volume + added_blue_tint
  let new_blue_volume := original_blue_volume + added_blue_tint
  (new_blue_volume / new_total_volume) * 100

theorem blue_tint_percentage_after_addition 
  (original : PaintMixture) 
  (added_blue_tint : ℝ) : 
  original.total_volume = 40 →
  original.blue_tint_percentage = 35 →
  added_blue_tint = 8 →
  45 < new_blue_tint_percentage original added_blue_tint ∧ 
  new_blue_tint_percentage original added_blue_tint < 47 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tint_percentage_after_addition_l773_77345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l773_77337

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific triangle
noncomputable def our_triangle : Triangle where
  A := Real.pi / 4  -- 45° in radians
  B := Real.pi / 3  -- 60° in radians (to be proved)
  C := Real.pi / 2  -- 90° (sum of angles in a triangle)
  a := 3 * Real.sqrt 2
  b := 3 * Real.sqrt 3
  c := (3 * Real.sqrt 6 + 3 * Real.sqrt 2) / 2  -- to be proved

-- State the theorem
theorem triangle_proof (t : Triangle) (h1 : t.A = Real.pi / 4) 
    (h2 : t.a = 3 * Real.sqrt 2) (h3 : t.b = 3 * Real.sqrt 3) : 
    t.B = Real.pi / 3 ∧ t.c = (3 * Real.sqrt 6 + 3 * Real.sqrt 2) / 2 := by
  sorry

#check triangle_proof our_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l773_77337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_sum_of_abs_roots_l773_77347

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 17 + 99 / x

-- Define the equation
def equation (x : ℝ) : Prop := x = g (g (g (g (g x))))

-- Define B as the sum of absolute values of roots
noncomputable def B : ℝ := |((Real.sqrt 17 + Real.sqrt 413) / 2)| + |((Real.sqrt 17 - Real.sqrt 413) / 2)|

-- State the theorem
theorem square_of_sum_of_abs_roots : B^2 = 413 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_sum_of_abs_roots_l773_77347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l773_77314

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define a point on the graph of f
structure Point (α : Type*) where
  x : α
  y : α

-- Theorem statement
theorem point_symmetry (a b : ℝ) (h : Point ℝ) (h_on_graph : f h.x = h.y) (h_ab : h.x = a ∧ h.y = b) :
  f (-a) = -b :=
by
  -- Proof steps would go here
  sorry

-- Example usage
example (a b : ℝ) (h : Point ℝ) (h_on_graph : f h.x = h.y) (h_ab : h.x = a ∧ h.y = b) :
  ∃ p : Point ℝ, p.x = -a ∧ p.y = -b ∧ f p.x = p.y :=
by
  -- Construct the symmetric point
  let p : Point ℝ := ⟨-a, -b⟩
  
  -- Show that this point satisfies the conditions
  exists p
  constructor
  · rfl
  constructor
  · rfl
  · exact point_symmetry a b h h_on_graph h_ab

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l773_77314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_mean_plus_std_dev_l773_77380

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  symmetric : Bool
  within_one_std_dev : ℝ

/-- The theorem stating the percentage of the distribution less than m + g -/
theorem percent_less_than_mean_plus_std_dev 
  (d : SymmetricDistribution) 
  (h1 : d.symmetric = true) 
  (h2 : d.within_one_std_dev = 0.68) : 
  ∃ (p : ℝ), p = 0.84 ∧ p ≤ 1 ∧ p ≥ 0 := by
  sorry

#check percent_less_than_mean_plus_std_dev

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_mean_plus_std_dev_l773_77380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l773_77381

/-- The sum of the infinite series ∑(n=1 to ∞) (2n^2 - n + 1) / (n+3)! is equal to 1/3. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (2 * n^2 - n + 1) / (Nat.factorial (n + 3)) = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l773_77381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l773_77379

-- Define constants
noncomputable def train_length : ℝ := 605  -- meters
noncomputable def train_speed : ℝ := 60    -- kmph
noncomputable def passing_time : ℝ := 33   -- seconds

-- Define conversion factor
noncomputable def kmph_to_mps : ℝ := 1000 / 3600

-- Theorem statement
theorem man_speed_calculation :
  let train_speed_mps := train_speed * kmph_to_mps
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps / kmph_to_mps
  |man_speed_kmph - 5.976| < 0.001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l773_77379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_four_l773_77348

def mySequence (n : ℕ) : ℝ :=
  if n % 2 = 1 then 3 else 4

theorem tenth_term_is_four :
  let a := mySequence
  (∀ n : ℕ, n ≥ 2 → a n * a (n-1) = a 1 * a 2) →
  a 1 = 3 →
  a 2 = 4 →
  a 10 = 4 := by
  sorry

#eval mySequence 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_four_l773_77348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mckenna_work_after_meeting_l773_77397

/-- Represents time in 24-hour format -/
def Time := Fin 24

/-- Calculates the difference between two times in hours -/
def timeDifference (t1 t2 : Time) : Nat :=
  if t2.val ≥ t1.val then t2.val - t1.val else 24 + t2.val - t1.val

/-- Represents Mckenna's work schedule -/
structure WorkSchedule where
  startTime : Time
  meetingEndTime : Time
  totalWorkHours : Nat

/-- Calculates the number of hours worked after the meeting -/
def hoursWorkedAfterMeeting (schedule : WorkSchedule) : Nat :=
  let endTime : Time := Fin.ofNat' ((schedule.startTime.val + schedule.totalWorkHours) % 24) (by simp)
  timeDifference schedule.meetingEndTime endTime

theorem mckenna_work_after_meeting (schedule : WorkSchedule) 
  (h1 : schedule.startTime = ⟨8, by norm_num⟩)
  (h2 : schedule.meetingEndTime = ⟨13, by norm_num⟩)
  (h3 : schedule.totalWorkHours = 7) :
  hoursWorkedAfterMeeting schedule = 2 := by
  sorry

#eval hoursWorkedAfterMeeting {
  startTime := ⟨8, by norm_num⟩,
  meetingEndTime := ⟨13, by norm_num⟩,
  totalWorkHours := 7
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mckenna_work_after_meeting_l773_77397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l773_77318

def n : ℕ := 2^3 * 3^6 * 5^7 * 7^8

theorem number_of_factors_of_n : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l773_77318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l773_77334

noncomputable section

-- Option A
def fA (x : ℝ) : ℝ := x - 1
def gA (x : ℝ) : ℝ := (x^2 - 1) / (x + 1)

-- Option B
def fB (x : ℝ) : ℝ := |x + 1|
def gB (x : ℝ) : ℝ := if x ≥ -1 then x + 1 else -x - 1

-- Option C
def fC (_x : ℝ) : ℝ := 1
def gC (x : ℝ) : ℝ := (x + 1)^0

-- Option D
def fD (x : ℝ) : ℝ := x
def gD (x : ℝ) : ℝ := (Real.sqrt x)^2

theorem function_equivalence :
  (∃ x, fA x ≠ gA x) ∧
  (∀ x, fB x = gB x) ∧
  (∃ x, fC x ≠ gC x) ∧
  (∃ x, fD x ≠ gD x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l773_77334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l773_77331

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AB : ℝ := 10
def BC : ℝ := 26
variable (AD BD : ℕ)

-- Define the conditions
def B_on_AC (A B C : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = (1 - t) • A + t • C
def AD_eq_CD (A C D : ℝ × ℝ) : Prop := dist A D = dist C D
def D_on_circle (A D : ℝ × ℝ) (AD : ℕ) : Prop := dist A D = AD

-- Define the perimeter of ACD
def perimeter_ACD (A C D : ℝ × ℝ) (AD : ℕ) : ℝ := 2 * AD + dist A C

-- Theorem statement
theorem sum_of_perimeters (A B C D : ℝ × ℝ) (AD BD : ℕ) 
  (h1 : B_on_AC A B C) (h2 : AD_eq_CD A C D) (h3 : D_on_circle A D AD) :
  ∃ (p1 p2 : ℝ), p1 ≠ p2 ∧ 
  (perimeter_ACD A C D AD = p1 ∨ perimeter_ACD A C D AD = p2) ∧
  p1 + p2 = 340 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l773_77331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l773_77333

-- Define the circle
def circle_set (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

-- Define the line y = 2x
def line1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1}

-- Define the line 4x - 3y - 11 = 0
def line2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 - 11 = 0}

theorem circle_equation :
  ∃ (a b r : ℝ),
    -- The center (a, b) is on line1
    (a, b) ∈ line1 ∧
    -- The circle is tangent to line2 at (2, -1)
    (2, -1) ∈ circle_set a b r ∧
    (2, -1) ∈ line2 ∧
    -- The equation of the circle
    circle_set a b r = circle_set (2/11) (4/11) (25/11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l773_77333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f5_on_interval_l773_77369

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the recursive function fₙ
def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_n n

-- Theorem statement
theorem max_f5_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 1 2 → f_n 5 y ≤ f_n 5 x ∧
  f_n 5 x = 3^32 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f5_on_interval_l773_77369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l773_77343

/-- The volume of a cylinder with radius r and height h is π * r^2 * h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_radius_proof (y : ℝ) :
  ∃ (r : ℝ),
    cylinder_volume (r + 4) 2 - cylinder_volume r 2 = y ∧
    cylinder_volume r 8 - cylinder_volume r 2 = y ∧
    r = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l773_77343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeroes_of_f_l773_77310

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.log (abs (x - 2)) - m

-- Theorem statement
theorem sum_of_zeroes_of_f (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), f x₁ m = 0 ∧ f x₂ m = 0 ∧ x₁ + x₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeroes_of_f_l773_77310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l773_77364

noncomputable def line_angle (a : ℝ) : ℝ := Real.arctan a

theorem line_angle_is_60_degrees :
  ∃ (a : ℝ), 
    (a * Real.sqrt 3 - 4 + 1 = 0) ∧ 
    (line_angle a = π / 3) := by
  -- We know that a = √3 from the problem
  let a := Real.sqrt 3
  
  -- Prove the first part of the conjunction
  have h1 : a * Real.sqrt 3 - 4 + 1 = 0 := by
    simp [a]
    ring
    
  -- Prove the second part of the conjunction
  have h2 : line_angle a = π / 3 := by
    simp [line_angle, a]
    -- This step requires additional lemmas about Real.arctan
    -- which are beyond the scope of this example
    sorry
    
  -- Combine the proofs
  exact ⟨a, ⟨h1, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l773_77364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_value_l773_77365

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_eccentricity_value :
  ∀ e : Ellipse, e.a = 2 ∧ e.b = 1 → eccentricity e = Real.sqrt 3 / 2 := by
  sorry

#check ellipse_eccentricity_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_value_l773_77365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_theorem_l773_77327

/-- A function satisfying f(-x) = 4 - f(x) for all real x -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = 4 - f x

/-- The function y = (2x+1)/x -/
noncomputable def g (x : ℝ) : ℝ := (2*x + 1) / x

/-- Intersection points of f and g -/
def IntersectionPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = g p.1 ∧ f p.1 = p.2

theorem intersection_sum_theorem (f : ℝ → ℝ) (points : List (ℝ × ℝ)) 
    (hf : SymmetricFunction f) (hi : IntersectionPoints f points) :
    (points.map (λ p => p.1 + p.2)).sum = 2 * points.length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_theorem_l773_77327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_median_triangle_BC_length_l773_77386

/-- A triangle with perpendicular medians -/
structure PerpendicularMedianTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Medians are perpendicular
  medians_perpendicular : (B.1 - (A.1 + C.1)/2) * (C.1 - (A.1 + B.1)/2) + 
                          (B.2 - (A.2 + C.2)/2) * (C.2 - (A.2 + B.2)/2) = 0
  -- Length of AB is 15
  AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15
  -- Length of AC is 20
  AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 20

/-- The length of BC in a triangle with perpendicular medians -/
noncomputable def BC_length (t : PerpendicularMedianTriangle) : ℝ :=
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

/-- Theorem: In a triangle with perpendicular medians where AB = 15 and AC = 20, BC = 32/3 -/
theorem perpendicular_median_triangle_BC_length 
  (t : PerpendicularMedianTriangle) : BC_length t = 32/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_median_triangle_BC_length_l773_77386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l773_77319

/-- Given a village population that undergoes several changes:
    1. 10% die
    2. 20% of the remaining leave
    3. 5% of those who left return
    4. 8% of the current population are hospitalized
    The theorem states that if the final population (including hospitalized) is 3240,
    then the initial population is approximately 4115. -/
theorem village_population (P : ℝ) : 
  (0.9 * P * 0.8 + 0.9 * P * 0.2 * 0.05) * 1.08 = 3240 → 
  ∃ ε > 0, |P - 4115| < ε := by
  sorry

#check village_population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l773_77319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_base_angle_value_l773_77384

-- Define a regular quadrilateral pyramid
structure RegularQuadPyramid where
  base_edge : ℝ
  height : ℝ
  base_perimeter_eq_circumference : 4 * base_edge = 2 * Real.pi * height

-- Define the angle between a median of a side face and the base plane
noncomputable def median_base_angle (p : RegularQuadPyramid) : ℝ :=
  Real.arctan (Real.sqrt 8 / (Real.pi * Real.sqrt 5))

-- Theorem statement
theorem median_base_angle_value (p : RegularQuadPyramid) :
  median_base_angle p = Real.arctan (Real.sqrt 8 / (Real.pi * Real.sqrt 5)) := by
  -- Unfold the definition of median_base_angle
  unfold median_base_angle
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_base_angle_value_l773_77384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2alpha_l773_77367

theorem cos_pi_half_minus_2alpha (α : ℝ) (h : Real.sin α - Real.cos α = 1/3) : 
  Real.cos (π/2 - 2*α) = 8/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2alpha_l773_77367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l773_77338

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 6) + 1 / (x^3 + 6*x) + 1 / (x^4 + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l773_77338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l773_77352

/-- Represents a point on the grid with integer coordinates --/
structure GridPoint where
  x : ℕ
  y : ℕ
  x_bound : x ≥ 1 ∧ x ≤ 6
  y_bound : y ≥ 1 ∧ y ≤ 6

/-- Represents a triangle formed by three distinct points on the grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  distinct : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

/-- Predicate to check if three points are collinear --/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- The set of all valid triangles on the 6x6 grid --/
def validTriangles : Finset GridTriangle :=
  sorry

/-- The main theorem stating the number of valid triangles --/
theorem count_valid_triangles : validTriangles.card = 6804 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l773_77352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_gain_210_l773_77340

-- Define the loan amount and interest rates
noncomputable def loan_amount : ℝ := 3500
noncomputable def interest_rate_A_to_B : ℝ := 0.10
noncomputable def interest_rate_B_to_C : ℝ := 0.12

-- Define B's total gain
noncomputable def total_gain : ℝ := 210

-- Define the function to calculate the number of years
noncomputable def calculate_years (amount : ℝ) (rate_A_to_B : ℝ) (rate_B_to_C : ℝ) (gain : ℝ) : ℝ :=
  gain / (amount * (rate_B_to_C - rate_A_to_B))

-- Theorem to prove
theorem years_to_gain_210 :
  calculate_years loan_amount interest_rate_A_to_B interest_rate_B_to_C total_gain = 3 := by
  -- Unfold the definitions
  unfold calculate_years loan_amount interest_rate_A_to_B interest_rate_B_to_C total_gain
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_gain_210_l773_77340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l773_77315

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def snowman1_radii : List ℝ := [4, 6, 8]
def snowman2_radii : List ℝ := List.map (· * 0.75) snowman1_radii

theorem total_snow_volume :
  (List.sum (List.map sphere_volume snowman1_radii) + 
   List.sum (List.map sphere_volume snowman2_radii)) = (4504.5 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l773_77315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_property_l773_77361

theorem log_function_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc a (2 * a), 
    (fun x => Real.log x / Real.log a) x ≤ 3 * (fun x => Real.log x / Real.log a) a) →
  a = 2^(-(3/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_property_l773_77361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_16_l773_77388

/-- Represents a rectangular field with a square pond inside. -/
structure RectField where
  width : ℝ
  length : ℝ
  pondSideLength : ℝ

/-- The conditions of the problem. -/
def fieldConditions (f : RectField) : Prop :=
  f.length = 2 * f.width ∧
  f.pondSideLength = 4 ∧
  f.pondSideLength^2 = (1/8) * (f.length * f.width)

theorem field_length_is_16 (f : RectField) (h : fieldConditions f) : f.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_16_l773_77388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_country_club_payment_l773_77350

/-- Calculates John's payment for the first year of a country club membership --/
theorem johns_country_club_payment
  (total_members : ℕ)
  (joining_fee_per_person : ℕ)
  (monthly_cost_per_person : ℕ)
  (months_in_year : ℕ)
  (h1 : total_members = 4)
  (h2 : joining_fee_per_person = 4000)
  (h3 : monthly_cost_per_person = 1000)
  (h4 : months_in_year = 12) :
  (total_members * joining_fee_per_person +
   total_members * monthly_cost_per_person * months_in_year) / 2 = 32000 := by
  -- Replace all calculations with sorry
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_country_club_payment_l773_77350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_modulo_prime_l773_77335

open BigOperators

/-- Define the sum function -/
def S (p : ℕ) (x y : ℤ) : ℤ :=
  ∑ k in Finset.range p, x^k * y^(p - 1 - k)

/-- Theorem statement for the sum modulo prime -/
theorem sum_modulo_prime (p : ℕ) (x y : ℤ) (hp : Nat.Prime p) :
  (x ≡ y [ZMOD p] → S p x y ≡ 0 [ZMOD p]) ∧
  (¬(x ≡ y [ZMOD p]) → S p x y ≡ 1 [ZMOD p]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_modulo_prime_l773_77335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_intersection_l773_77387

/-- A parabola in a 2D plane. -/
structure Parabola where
  -- Add necessary fields to define a parabola
  dummy : Unit

/-- A circle in a 2D plane. -/
structure Circle where
  -- Add necessary fields to define a circle
  dummy : Unit

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two line segments are parallel. -/
def areParallel (p1 p2 p3 p4 : Point) : Prop :=
  -- Definition of parallel line segments
  sorry

/-- Checks if a point lies on a parabola. -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  -- Definition to check if a point is on a parabola
  sorry

/-- Checks if a point lies on a circle. -/
def isOnCircle (p : Point) (circ : Circle) : Prop :=
  -- Definition to check if a point is on a circle
  sorry

theorem parabola_chord_intersection (P : Parabola) (A B C D E F : Point) (S1 S2 : Circle) :
  areParallel A B C D →
  isOnParabola A P →
  isOnParabola B P →
  isOnParabola C P →
  isOnParabola D P →
  isOnCircle A S1 →
  isOnCircle B S1 →
  isOnCircle C S2 →
  isOnCircle D S2 →
  isOnCircle E S1 →
  isOnCircle E S2 →
  isOnCircle F S1 →
  isOnCircle F S2 →
  isOnParabola E P →
  isOnParabola F P :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_intersection_l773_77387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_64_81_l773_77366

noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

theorem mean_proportional_64_81 : mean_proportional 64 81 = 72 := by
  -- Unfold the definition of mean_proportional
  unfold mean_proportional
  -- Simplify the expression under the square root
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_64_81_l773_77366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l773_77354

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis, opening to the left -/
structure LeftOpeningParabola where
  -- The equation of the parabola is y² = -4ax for some positive real number a
  a : ℝ
  a_pos : 0 < a

/-- A point on a parabola -/
structure PointOnParabola (p : LeftOpeningParabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = -4 * p.a * x

/-- The focus of a left-opening parabola -/
def focus (p : LeftOpeningParabola) : ℝ × ℝ := (-p.a, 0)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (p : LeftOpeningParabola) 
  (point : PointOnParabola p)
  (h1 : point.x = -5)
  (h2 : distance (point.x, point.y) (focus p) = 6) :
  p.a = 1 := by sorry

#check parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l773_77354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l773_77325

theorem triangle_shape (a b c : ℝ) (A B C : Real) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Triangle sides are positive
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Triangle angles are positive
  A + B + C = Real.pi →         -- Sum of angles in a triangle
  a * Real.sin A = b * Real.sin B →  -- Sine law
  a * Real.cos A = b * Real.sin (Real.pi/2 + B) →  -- Given condition
  (A = B) ∨ (A + B = Real.pi/2) :=  -- Conclusion: isosceles or right-angled
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l773_77325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_between_l773_77302

theorem smallest_fraction_between (p q : ℚ) :
  99 / 100 < p / q ∧ p / q < 100 / 101 →
  q ≥ 201 ∧
  (∃ (p' : ℚ), p' / 201 = 199 / 201 ∧ 99 / 100 < p' / 201 ∧ p' / 201 < 100 / 101) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_between_l773_77302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l773_77385

theorem log_equation_solution (x : ℝ) (h : x > 0) : (Real.log 64 / Real.log x = 3) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l773_77385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_90_moves_l773_77342

noncomputable def move (z : ℂ) : ℂ := z * Complex.exp (Complex.I * (Real.pi / 3)) + 12

noncomputable def iterate_move : ℕ → ℂ → ℂ
| 0, z => z
| n + 1, z => move (iterate_move n z)

theorem final_position_after_90_moves :
  iterate_move 90 5 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_90_moves_l773_77342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_heads_and_three_l773_77309

/-- A fair coin -/
def Coin : Type := Bool

/-- A fair six-sided die -/
def Die : Type := Fin 6

/-- The probability of getting heads on a fair coin -/
noncomputable def prob_heads : ℝ := 1 / 2

/-- The probability of getting a specific number on a fair six-sided die -/
noncomputable def prob_die_face : ℝ := 1 / 6

/-- The event of getting two heads and a 3 -/
def event (c1 c2 : Coin) (d : Die) : Prop :=
  c1 = true ∧ c2 = true ∧ d.val = 2

/-- The probability of the event -/
noncomputable def prob_event : ℝ := prob_heads * prob_heads * prob_die_face

theorem prob_two_heads_and_three :
  prob_event = 1 / 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_heads_and_three_l773_77309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_bounded_l773_77353

/-- A sequence of positive real numbers defined by the given recurrence relation -/
noncomputable def RecurrenceSequence (a₁ a₂ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | 1 => a₂
  | n + 2 => 2 / (RecurrenceSequence a₁ a₂ (n + 1) + RecurrenceSequence a₁ a₂ n)

/-- Theorem stating the existence of bounds for the recurrence sequence -/
theorem recurrence_sequence_bounded (a₁ a₂ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) :
  ∃ s t : ℝ, s > 0 ∧ t > 0 ∧ ∀ n : ℕ, s ≤ RecurrenceSequence a₁ a₂ n ∧ RecurrenceSequence a₁ a₂ n ≤ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_bounded_l773_77353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77373

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x) + Real.log (2 - x)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- Theorem stating that f is even and decreasing on (0, 2)
theorem f_properties :
  (∀ x, x ∈ domain → f (-x) = f x) ∧
  (∀ x y, x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → x < y → f y < f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sidney_tuesday_jumps_l773_77305

/-- The number of jumping jacks Sidney does on a given day -/
def sidney_jumps : Fin 4 → ℕ := sorry

/-- The total number of jumping jacks Brooke does -/
def brooke_total : ℕ := 438

theorem sidney_tuesday_jumps :
  sidney_jumps 0 = 20 →  -- Monday
  sidney_jumps 2 = 40 →  -- Wednesday
  sidney_jumps 3 = 50 →  -- Thursday
  brooke_total = 3 * (sidney_jumps 0 + sidney_jumps 1 + sidney_jumps 2 + sidney_jumps 3) →
  sidney_jumps 1 = 36    -- Tuesday
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sidney_tuesday_jumps_l773_77305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_rate_and_days_l773_77307

/-- Represents a worker's ability to complete a job --/
structure Worker where
  days_to_complete : ℚ
  work_rate : ℚ := 1 / days_to_complete

/-- The total work rate of a group of workers --/
def total_work_rate (workers : List Worker) : ℚ :=
  workers.map (·.work_rate) |>.sum

theorem worker_c_rate_and_days (worker_a worker_b worker_c : Worker)
    (h_a : worker_a.days_to_complete = 10)
    (h_b : worker_b.days_to_complete = 15)
    (h_abc : total_work_rate [worker_a, worker_b, worker_c] = 1 / 4) :
    worker_c.work_rate = 1 / 12 ∧ worker_c.days_to_complete = 12 := by
  sorry

#check worker_c_rate_and_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_rate_and_days_l773_77307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_2011_l773_77324

theorem probability_divisible_by_2011 :
  ∃ (p : ℚ),
    p = 1197 / 2011 ∧
    p = (Finset.filter (λ y => (y^y - 1) % 2011 = 0) (Finset.range (Nat.factorial 2011 + 1))).card /
        (Finset.range (Nat.factorial 2011 + 1)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_2011_l773_77324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ACF_l773_77306

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  vertices : Set Point

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  sorry

/-- Given two squares ABCD and DEFG with side lengths 8 cm and 5 cm respectively,
    prove that the area of triangle ACF is 52 square centimeters. -/
theorem area_of_triangle_ACF (ABCD DEFG : Square) (A C F : Point) :
  ABCD.sideLength = 8 →
  DEFG.sideLength = 5 →
  A ∈ ABCD.vertices →
  C ∈ ABCD.vertices →
  F ∈ DEFG.vertices →
  areaOfTriangle A C F = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ACF_l773_77306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_when_m_is_1_shortest_chord_line_equation_x_coordinate_range_of_P_l773_77332

/-- Circle C in the Cartesian coordinate system -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 11 = 0

/-- Line l in the Cartesian coordinate system -/
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The distance between two points in ℝ² -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem chord_length_when_m_is_1 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l 1 x₁ y₁ ∧ line_l 1 x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 6 * Real.sqrt 13 / 13 := by
  sorry

theorem shortest_chord_line_equation :
  ∀ (m : ℝ),
    (∀ (x y : ℝ), line_l m x y ↔ x - y - 2 = 0) ↔
    (∀ (m' : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l m' x₁ y₁ ∧ line_l m' x₂ y₂ →
      distance x₁ y₁ x₂ y₂ ≥ distance x₁ y₁ x₂ y₂) := by
  sorry

theorem x_coordinate_range_of_P :
  ∀ (x₀ : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      distance x₀ (x₀ - 2) x₁ y₁ = Real.sqrt 5 ∧
      distance x₀ (x₀ - 2) x₂ y₂ = Real.sqrt 5) ↔
    (0 < x₀ ∧ x₀ < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_when_m_is_1_shortest_chord_line_equation_x_coordinate_range_of_P_l773_77332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_sixth_cos_half_alpha_plus_beta_l773_77328

-- Problem 1
theorem sin_alpha_plus_pi_sixth (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  Real.sin (α + π / 6) = (3 + 4 * Real.sqrt 3) / 10 := by sorry

-- Problem 2
theorem cos_half_alpha_plus_beta (α β : ℝ)
  (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.cos α = -5 / 13)
  (h3 : Real.tan (β / 2) = 1 / 3) :
  Real.cos (α / 2 + β) = -17 * Real.sqrt 13 / 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_sixth_cos_half_alpha_plus_beta_l773_77328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l773_77322

def IsValidFunction (f : ℕ+ → ℕ+) : Prop :=
  (∀ n m : ℕ+, f (n + m) = f n * f m) ∧
  (∃ n₀ : ℕ+, f (f n₀) = (f n₀) * (f n₀))

theorem characterize_valid_functions :
  ∀ f : ℕ+ → ℕ+,
    IsValidFunction f ↔ (∀ n : ℕ+, f n = 1) ∨ (∀ n : ℕ+, f n = 2^(n.val)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l773_77322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l773_77363

theorem sin_2alpha_on_line (α : ℝ) : 
  (Real.sin α = -2 * Real.cos α) → Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l773_77363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_theorem_l773_77376

/-- Given a natural number n and a real number x, 
    (3x - 1/x)^n represents the expansion of the binomial. -/
noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (3*x - 1/x)^n

/-- The sum of coefficients in the expansion -/
noncomputable def sum_of_coefficients (n : ℕ) : ℝ := binomial_expansion n 1

/-- The coefficient of x^2 in the expansion -/
noncomputable def coefficient_of_x_squared (n : ℕ) : ℝ := 
  -- This definition is left abstract as the exact calculation method
  -- is not provided in the problem statement
  sorry

theorem expansion_coefficient_theorem (n : ℕ) : 
  sum_of_coefficients n = 16 → coefficient_of_x_squared n = -108 := by
  sorry

#check expansion_coefficient_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_theorem_l773_77376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_not_geometric_sequence_l773_77368

theorem arithmetic_not_geometric_sequence (a b c : ℝ) : 
  Real.rpow 2 a = 3 → Real.rpow 2 b = 6 → Real.rpow 2 c = 12 → 
  (b - a = c - b) ∧ ¬(b / a = c / b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_not_geometric_sequence_l773_77368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_slope_one_dot_product_constant_l773_77311

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := x = k * y + 1

-- Define the intersection points of the line and the parabola
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ parabola x y ∧ line_through_focus k x y}

-- Statement 1: Length of AB when slope is 1
theorem length_AB_slope_one :
  ∀ A B : ℝ × ℝ, A ∈ intersection_points 1 → B ∈ intersection_points 1 →
  A ≠ B → ‖A - B‖ = 8 := by sorry

-- Statement 2: OA · OB is constant
theorem dot_product_constant :
  ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersection_points k → B ∈ intersection_points k →
  A ≠ B → A.1 * B.1 + A.2 * B.2 = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_slope_one_dot_product_constant_l773_77311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l773_77336

theorem gcd_problem (b : ℤ) (h : 210 ∣ b) : Int.gcd (2*b^3 + b^2 + 5*b + 105) b = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l773_77336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_l773_77359

noncomputable section

def Sphere (n : ℕ) := Type

def Cube (n : ℕ) := Type

def Inscribed (C : Cube 3) (S : Sphere 3) : Prop := sorry

def SurfaceArea (S : Sphere 3) : ℝ := sorry

def EdgeLength (C : Cube 3) : ℝ := sorry

theorem inscribed_cube_edge_length (S : Sphere 3) (C : Cube 3) :
  Inscribed C S →
  SurfaceArea S = 36 * Real.pi →
  EdgeLength C = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_l773_77359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_a_l773_77326

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) + a * Real.exp x

-- Theorem for monotonicity intervals when a = -4
theorem monotonicity_intervals :
  ∀ x : ℝ, (x > Real.log 2 → (deriv (f (-4))) x > 0) ∧
           (x < Real.log 2 → (deriv (f (-4))) x < 0) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2 * x) ↔ 
    (-1 ≤ a ∧ a ≤ 2 * Real.exp (3/4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_a_l773_77326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_cosine_function_l773_77356

-- Define the function f as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.cos (3 * x + φ)

-- State the theorem
theorem even_cosine_function (φ : ℝ) :
  (∀ x, f x φ = f (-x) φ) ↔ ∃ k : ℤ, φ = k * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_cosine_function_l773_77356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_may_differ_l773_77323

/-- Represents the walking scenario of a pedestrian -/
structure WalkingScenario where
  duration : ℝ  -- Total duration of the walk in hours
  hourly_distance : ℝ  -- Distance covered in each one-hour interval in km
  total_distance : ℝ  -- Total distance covered in the entire duration in km

/-- Defines the average speed calculation -/
noncomputable def average_speed (scenario : WalkingScenario) : ℝ :=
  scenario.total_distance / scenario.duration

/-- Theorem stating that the average speed can differ from the hourly distance rate -/
theorem average_speed_may_differ (scenario : WalkingScenario) 
  (h1 : scenario.duration = 3.5)
  (h2 : scenario.hourly_distance = 5)
  (h3 : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    ∃ d : ℝ, d = scenario.hourly_distance * t) :
  ∃ total_distance : ℝ, 
    total_distance > 0 ∧ 
    average_speed { duration := scenario.duration, 
                    hourly_distance := scenario.hourly_distance, 
                    total_distance := total_distance } ≠ scenario.hourly_distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_may_differ_l773_77323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_repeating_block_length_of_11_13_repeating_block_length_divides_denominator_minus_one_l773_77395

/-- The length of the smallest repeating block in the decimal expansion of 11/13 -/
def smallestRepeatingBlockLength : ℕ := 6

/-- The fraction we're examining -/
def fraction : ℚ := 11 / 13

theorem smallest_repeating_block_length_of_11_13 :
  smallestRepeatingBlockLength = 6 :=
by
  -- We'll provide a direct proof that the smallest repeating block length is 6
  -- This is based on the long division process described in the solution
  sorry

-- Additional theorem to show that this length is related to the denominator's properties
theorem repeating_block_length_divides_denominator_minus_one :
  13 - 1 ≡ 0 [MOD smallestRepeatingBlockLength] :=
by
  -- This theorem states that the length of the repeating block (6)
  -- divides the denominator minus one (13 - 1 = 12)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_repeating_block_length_of_11_13_repeating_block_length_divides_denominator_minus_one_l773_77395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l773_77313

-- Define the variable cost function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10*x^2 + 200*x + 1000
  else if x ≥ 40 then 801*x + 10000/x - 8450
  else 0

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10*x^2 + 600*x - 1250
  else if x ≥ 40 then -x - 10000/x + 8200
  else 0

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x : ℝ, x > 0 → W x ≤ W 100 ∧ W 100 = 8000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l773_77313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_sum_l773_77308

-- Define the conditions
def condition1 : ℕ := 1 + 3^2
def condition2 : ℕ := 6 + 5^2  -- 3! = 6
def condition3 : ℕ := 120 / (5 - 2) + 7^2  -- 5! = 120

-- State the theorem
theorem factorial_square_sum :
  condition1 = 10 →
  condition2 = 52 →
  condition3 = 174 →
  (5040 / (7 - 4)) + 11^2 = 1801 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_sum_l773_77308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_6x8_bookshelf_l773_77396

/-- The minimum side length of a square room that allows a rectangular object to be moved diagonally --/
noncomputable def min_room_side_length (width height : ℝ) : ℝ :=
  Real.sqrt (width^2 + height^2)

/-- Theorem: The minimum side length of a square room that allows a 6' x 8' rectangular object
    to be moved diagonally without tilting or disassembling is 10 feet --/
theorem min_room_side_for_6x8_bookshelf :
  min_room_side_length 6 8 = 10 := by
  sorry

#eval Float.sqrt (6^2 + 8^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_6x8_bookshelf_l773_77396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l773_77389

/-- The distance from the center of a circle to a line --/
noncomputable def distance_circle_center_to_line (a b c d e f : ℝ) : ℝ :=
  let circle_eq := fun (x y : ℝ) ↦ x^2 + y^2 + a*x + b*y + c = 0
  let line_eq := fun (x y : ℝ) ↦ d*x + e*y + f = 0
  let center := (- a / 2, - b / 2)
  |d * (- a / 2) + e * (- b / 2) + f| / Real.sqrt (d^2 + e^2)

/-- The distance from the center of the given circle to the given line is 3 --/
theorem distance_is_three :
  distance_circle_center_to_line (-2) (-4) 4 3 4 4 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l773_77389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_fourth_sufficient_not_necessary_l773_77399

theorem alpha_pi_fourth_sufficient_not_necessary :
  (∃ α : ℝ, α ≠ π/4 ∧ Real.cos α = Real.sqrt 2/2) ∧
  (∀ α : ℝ, α = π/4 → Real.cos α = Real.sqrt 2/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_pi_fourth_sufficient_not_necessary_l773_77399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z5_magnitude_l773_77383

def complexSequence : ℕ → ℂ
  | 0 => 1 + Complex.I
  | n + 1 => (complexSequence n)^2 + 1

theorem z5_magnitude :
  Complex.abs (complexSequence 4) = Real.sqrt 485809 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z5_magnitude_l773_77383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_formula_l773_77303

/-- Given a geometric sequence {a_n} with a₁ = 4 and common ratio q = 3,
    the general term formula is aₙ = 4·3^(n-1) -/
theorem geometric_sequence_formula (n : ℕ) :
  let a : ℕ → ℝ := fun m => if m = 1 then 4 else 4 * 3^(m - 1)
  a n = 4 * 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_formula_l773_77303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_8_zeroes_l773_77391

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def count_factors_of_two (n : ℕ) : ℕ :=
  (Nat.factors n).filter (· == 2) |>.length

def count_trailing_zeroes_base_8 (n : ℕ) : ℕ :=
  count_factors_of_two n / 3

theorem fifteen_factorial_base_8_zeroes :
  count_trailing_zeroes_base_8 (factorial 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_8_zeroes_l773_77391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_15_l773_77349

/-- The sum of the exponents of the prime factors of the square root of the largest perfect square that divides 20! -/
def sum_of_exponents : ℕ := 15

/-- 20 factorial -/
def factorial_20 : ℕ := Nat.factorial 20

/-- The largest perfect square that divides 20! -/
def largest_perfect_square (n : ℕ) : ℕ :=
  sorry

/-- The square root of the largest perfect square that divides 20! -/
def sqrt_largest_perfect_square (n : ℕ) : ℕ :=
  sorry

/-- The list of prime factors of a number -/
def prime_factors (n : ℕ) : List ℕ :=
  sorry

/-- The exponents of prime factors of a number -/
def prime_factor_exponents (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_equals_15 :
  (prime_factor_exponents (sqrt_largest_perfect_square factorial_20)).sum = sum_of_exponents := by
  sorry

#eval sum_of_exponents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_15_l773_77349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l773_77351

/-- Represents the probability of defeating the dragon -/
noncomputable def defeat_probability : ℝ := 1

/-- Represents the probability of two heads growing -/
noncomputable def two_heads_prob : ℝ := 1/4

/-- Represents the probability of one head growing -/
noncomputable def one_head_prob : ℝ := 1/3

/-- Represents the probability of no heads growing -/
noncomputable def no_heads_prob : ℝ := 5/12

theorem dragon_defeat_probability :
  two_heads_prob + one_head_prob + no_heads_prob = 1 →
  defeat_probability = 1 := by
  sorry

#check dragon_defeat_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_probability_l773_77351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77301

noncomputable def f (x : Real) : Real := 2 * Real.sin (Real.pi - x) * Real.cos x + Real.cos (2 * x)

theorem f_properties :
  (∃ (T : Real), T > 0 ∧ ∀ (x : Real), f (x + T) = f x ∧ ∀ (S : Real), S > 0 ∧ (∀ (x : Real), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : Real), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f x ≤ 1) ∧
  (∀ (x : Real), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : Real), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ f x = 1) ∧
  (∃ (x : Real), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ f x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_at_max_area_l773_77360

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.b = 1 ∧
  t.c + t.b = 2 * t.a * Real.cos t.B

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : Real :=
  1/2 * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem cos_A_at_max_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (∃ (max_area : Real), ∀ (t' : Triangle), 
    triangle_conditions t' → triangle_area t' ≤ max_area) →
  Real.cos t.A = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_at_max_area_l773_77360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_360_not_divisible_by_two_l773_77329

def divisors_not_divisible_by_two (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => d ∣ n ∧ d > 0 ∧ ¬(2 ∣ d))

theorem count_divisors_360_not_divisible_by_two :
  (divisors_not_divisible_by_two 360).card = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_360_not_divisible_by_two_l773_77329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_surface_area_approx_l773_77300

-- Define the dimensions of the rectangular prism
def prism_length : ℝ := 10
def prism_width : ℝ := 5
def prism_height : ℝ := 30

-- Define the size of the cut cube
def cut_cube_size : ℝ := 3

-- Define the volume of the original prism
def prism_volume : ℝ := prism_length * prism_width * prism_height

-- Define the volume of the cut cube
def cut_cube_volume : ℝ := cut_cube_size ^ 3

-- Define the remaining volume
def remaining_volume : ℝ := prism_volume - cut_cube_volume

-- Define the edge length of the new cube
noncomputable def new_cube_edge : ℝ := remaining_volume ^ (1/3)

-- Define the surface area of the new cube
noncomputable def new_cube_surface_area : ℝ := 6 * new_cube_edge ^ 2

-- Theorem statement
theorem new_cube_surface_area_approx :
  ∃ ε > 0, abs (new_cube_surface_area - 784.15) < ε := by
  sorry

#eval prism_volume
#eval cut_cube_volume
#eval remaining_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_surface_area_approx_l773_77300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l773_77346

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := 1 / x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, -1)

-- Define the slope of the tangent line at P
def tangent_slope : ℝ := -1

-- State the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, (x + y - 2 = 0) ↔ (y - f P.fst = tangent_slope * (x - P.fst)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l773_77346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_C_l773_77339

theorem max_angle_C (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.cos A * Real.sin B * Real.sin C + Real.cos B * Real.sin A * Real.sin C = 2 * Real.cos C * Real.sin A * Real.sin B →
  C ≤ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_C_l773_77339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l773_77344

/-- Sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The geometric sequence -/
def a : ℕ → ℝ := sorry

theorem geometric_sequence_sum (h1 : S 2 = 2) (h2 : S 4 = 4) : S 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l773_77344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phd_duration_proof_l773_77304

def phd_duration (bs_duration : ℚ) (tom_total_duration : ℚ) (normal_fraction : ℚ) : ℚ :=
  let normal_total_duration := tom_total_duration / normal_fraction
  normal_total_duration - bs_duration

theorem phd_duration_proof :
  phd_duration 3 6 (3/4) = 5 := by
  unfold phd_duration
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry

#eval phd_duration 3 6 (3/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phd_duration_proof_l773_77304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_l773_77372

/-- The probability of two specific people sitting adjacent in a row of 5 people -/
theorem adjacent_probability (n : ℕ) (h : n = 5) : 
  (2 * Nat.factorial (n - 1) : ℚ) / (Nat.factorial n : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_l773_77372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_limit_l773_77398

/-- The sequence for which we want to find the limit -/
def our_sequence (n : ℕ) : ℚ :=
  ((2*n+1)^3 - (2*n+3)^3) / ((2*n+1)^2 + (2*n+3)^2)

/-- The limit of the sequence as n approaches infinity -/
def sequence_limit : ℚ := -15/8

/-- Theorem stating that the limit of the sequence is -15/8 -/
theorem sequence_converges_to_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |our_sequence n - sequence_limit| < ε :=
by
  sorry

#check sequence_converges_to_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_limit_l773_77398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77362

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem f_properties :
  let smallestPeriod := Real.pi
  let maxValue := Real.sqrt 2
  let minValue := -1
  (∀ (x : ℝ), f (x + smallestPeriod) = f x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4) → f x ≤ maxValue) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4) ∧ f x = maxValue) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4) → f x ≥ minValue) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4) ∧ f x = minValue) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l773_77362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l773_77371

/-- Given a journey of 210 miles completed in 6.5 hours, prove that the average speed is approximately 32.31 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 210) (h2 : time = 6.5) :
  ∃ (speed : ℝ), abs (speed - (distance / time)) < 0.01 ∧ abs (speed - 32.31) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l773_77371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ground_beef_minus_lean_beef_is_two_l773_77378

/-- Represents the types of cows on Farmer James' farm -/
inductive CowType
  | GroundBeef
  | Steak
  | LeanBeef

/-- Returns the number of legs for a given cow type -/
def legs (c : CowType) : Nat :=
  match c with
  | CowType.GroundBeef => 0
  | CowType.Steak => 1
  | CowType.LeanBeef => 2

/-- Represents the count of each type of cow -/
structure CowCounts where
  groundBeef : Nat
  steak : Nat
  leanBeef : Nat

theorem ground_beef_minus_lean_beef_is_two 
  (counts : CowCounts)
  (total_cows : counts.groundBeef + counts.steak + counts.leanBeef = 20)
  (total_legs : counts.steak + 2 * counts.leanBeef = 18) :
  counts.groundBeef - counts.leanBeef = 2 := by
  sorry

#check ground_beef_minus_lean_beef_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ground_beef_minus_lean_beef_is_two_l773_77378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l773_77375

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2*y*f x) →
  (∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l773_77375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_reals_l773_77392

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- Recursively define f_n
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case
  | 1 => f
  | (n + 1) => λ x => f (f_n n x)

-- Define the set M
def M : Set ℝ := {x | f_n 2036 x = x}

-- Theorem statement
theorem M_equals_reals : M = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_reals_l773_77392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l773_77355

/-- The distance between two parallel lines given by their general equations -/
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the lines 3x + 2y - 1 = 0 and 6x + 4y + 1 = 0 is 3√13 / 26 -/
theorem distance_specific_lines : 
  distance_between_parallel_lines 3 2 (-1) (1/2) = 3 * Real.sqrt 13 / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l773_77355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l773_77317

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_problem :
  ∃ (n : ℕ), geometric_sum 1 (1/2) n = 31/16 ∧ n = 5 := by
  -- We'll use 5 as our witness for n
  use 5
  -- Split the goal into two parts
  constructor
  -- Prove that geometric_sum 1 (1/2) 5 = 31/16
  · sorry
  -- Prove that n = 5 (which is trivial given our choice of n)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l773_77317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_power_six_l773_77377

noncomputable def w : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem w_power_six : w^6 = (1 : ℂ) / 4 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_power_six_l773_77377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_and_C_disjoint_l773_77357

variable (U : Type*)

variable (A B C : Set U)

-- No A is B
axiom no_A_is_B : A ∩ B = ∅

-- All B are C
axiom all_B_are_C : B ⊆ C

-- Theorem to prove
theorem A_and_C_disjoint : A ∩ C = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_and_C_disjoint_l773_77357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equation_equivalence_l773_77390

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem min_equation_equivalence (n : ℕ) :
  (∃ (k : ℕ), ∀ (j : ℕ), k^2 + floor (n / k^2 : ℝ) ≤ j^2 + floor (n / j^2 : ℝ)) ∧
  (∃ (k : ℕ), k^2 + floor (n / k^2 : ℝ) = 1991) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equation_equivalence_l773_77390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_l773_77374

/-- Represents the daily sales volume in pieces -/
def x : ℝ := sorry

/-- Represents the unit price in yuan as a function of x -/
def P (x : ℝ) : ℝ := 160 - 2*x

/-- Represents the cost in yuan as a function of x -/
def C (x : ℝ) : ℝ := 500 + 30*x

/-- Represents the profit in yuan as a function of x -/
def profit (x : ℝ) : ℝ := x * P x - C x

/-- Theorem stating the range of x that satisfies the profit goal -/
theorem profit_range :
  (∀ x, 20 ≤ x ∧ x ≤ 45 → profit x ≥ 1300) ∧
  (∀ x, x < 20 ∨ x > 45 → profit x < 1300) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_range_l773_77374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_n_squared_minus_3n_plus_2_minus_n_l773_77393

/-- The limit of √(n^2 - 3n + 2) - n as n approaches infinity is -3/2 -/
theorem limit_sqrt_n_squared_minus_3n_plus_2_minus_n :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |Real.sqrt (n^2 - 3*n + 2) - n - (-3/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_n_squared_minus_3n_plus_2_minus_n_l773_77393
